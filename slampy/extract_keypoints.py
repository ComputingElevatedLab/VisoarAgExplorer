import cv2
import numpy as np

from OpenVisus          import *
from slampy.image_utils import *

# //////////////////////////////////////////////////////////////////////////
class ExtractKeyPoints:

	# constructor
	def __init__(self,min_num_keypoints,max_num_keypoints,anms,extractor_method="akaze", timing=False):

		self.extractor_method=extractor_method
		self.timing = timing
		
		if (extractor_method == "akaze"):
			self.detector = cv2.AKAZE.create()
			self.akaze_threshold=self.detector.getThreshold()
		elif (extractor_method == "kaze"):
			self.detector = cv2.KAZE.create()
			self.kaze_threshold=self.detector.getThreshold()
		elif (extractor_method == "sift"):
			# if we don't set a maximum number of features to find, 
			# it'll occasionally find an incomprehensibly large amount
			self.detector = cv2.SIFT.create(nfeatures=max_num_keypoints)
		elif (extractor_method == "orb"):
			self.detector = cv2.ORB.create()
			self.fastThreshold = self.detector.getFastThreshold()
			self.maxFeatures = max_num_keypoints
		else:
			print(f"Keypoint extraction method {extractor_method} not recognized, defaulting to akaze...")
			self.detector = cv2.AKAZE.create()
			self.extractor_method = "akaze"

		self.min_num_keypoints=min_num_keypoints
		self.max_num_keypoints=max_num_keypoints
		self.anms=anms

	# doExtract
	def doExtract(self, energy):
		if (self.extractor_method == "akaze"):
			return self.doExtractAkaze(energy)
		elif (self.extractor_method == "kaze"):
			return self.doExtractKaze(energy)
		elif (self.extractor_method == "sift"):
			return self.doExtractSift(energy)
		elif (self.extractor_method == "orb"):
			return self.doExtractOrb(energy)
		else:
			print(f"extractor_method {self.extractor_method} not recognized.")
			return None


	def doExtractAkaze(self, energy): 
		T1 = Time.now()

		#detect keypoints
		t2 = Time.now()
		keypoints=[]
		history=[]
		while True:
			if not self.akaze_threshold:
				self.akaze_threshold = .1
			self.detector.setThreshold(self.akaze_threshold)
			keypoints=self.detector.detect(energy)
			N = len(keypoints)
			print("akaze threshold " , self.akaze_threshold , " got " , N , " keypoints")

			# after 4 "zeros" assume the image is simply wrong
			history.append(N)
			if history[-4:]==[0]*4:
				print("Failed to extract keypoints")
				return ([],[])

			if self.min_num_keypoints>0.001 and N < self.min_num_keypoints :
				self.akaze_threshold *= 0.8
				continue

			if self.max_num_keypoints>0.001 and N > self.max_num_keypoints :
				self.akaze_threshold *= 1.2
				continue

			break

		msec_detect = t2.elapsedMsec()

		t2 = Time.now()
		if self.anms>0 and len(keypoints)>self.anms:

			# important!
			# sort keypoints in DESC order by response 
			# remember since anms need them to be in order
			keypoints=sorted(keypoints,key=lambda A: A.response,reverse=True)

			responses=[keypoint.response for keypoint in keypoints]
			xs=[keypoint.pt[0] for keypoint in keypoints]
			ys=[keypoint.pt[1] for keypoint in keypoints]
	
			good_indices = KeyPoint.adaptiveNonMaximalSuppression(responses,xs,ys,self.anms)

			keypoints=[keypoints[it] for it in good_indices]

		msec_anms=t2.elapsedMsec()

		# compute descriptors
		t2 = Time.now()
		keypoints,descriptors=self.detector.compute(energy, keypoints)
		msec_compute = t2.elapsedMsec()

		print("Extracted",len(keypoints), " keypoints"," in " , T1.elapsedMsec() ," msec", "msec_detect", msec_detect, "msec_compute" , msec_compute, "msec_anms",msec_anms)
		if self.timing:
			return keypoints, descriptors, T1.elapsedMsec() / 1000
		else:
			return keypoints, descriptors


	def doExtractKaze(self, energy): 
		T1 = Time.now()

		#detect keypoints
		t2 = Time.now()
		keypoints=[]
		history=[]
		while True:
			if not self.kaze_threshold:
				self.kaze_threshold = .1
			self.detector.setThreshold(self.kaze_threshold)
			keypoints=self.detector.detect(energy)
			N = len(keypoints)
			print("kaze threshold " , self.kaze_threshold , " got " , N , " keypoints")

			# after 4 "zeros" assume the image is simply wrong
			history.append(N)
			if history[-4:]==[0]*4:
				print("Failed to extract keypoints")
				return ([],[])

			if self.min_num_keypoints>0.001 and N < self.min_num_keypoints :
				self.kaze_threshold *= 0.8
				continue

			if self.max_num_keypoints>0.001 and N > self.max_num_keypoints :
				self.kaze_threshold *= 1.2
				continue

			break

		msec_detect = t2.elapsedMsec()

		t2 = Time.now()
		if self.anms>0 and len(keypoints)>self.anms:

			# important!
			# sort keypoints in DESC order by response 
			# remember since anms need them to be in order
			keypoints=sorted(keypoints,key=lambda A: A.response,reverse=True)

			responses=[keypoint.response for keypoint in keypoints]
			xs=[keypoint.pt[0] for keypoint in keypoints]
			ys=[keypoint.pt[1] for keypoint in keypoints]
	
			good_indices = KeyPoint.adaptiveNonMaximalSuppression(responses,xs,ys,self.anms)

			keypoints=[keypoints[it] for it in good_indices]

		msec_anms=t2.elapsedMsec()

		# compute descriptors
		t2 = Time.now()

		keypoints,descriptors=self.detector.compute(energy, keypoints)

		# convert descriptors to uint8
		descriptors = (descriptors * 255).astype('uint8')

		msec_compute = t2.elapsedMsec()

		print("Extracted",len(keypoints), " keypoints"," in " , T1.elapsedMsec() ," msec", "msec_detect", msec_detect, "msec_compute" , msec_compute, "msec_anms",msec_anms)
		if self.timing:
			return keypoints, descriptors, T1.elapsedMsec() / 1000
		else:
			return keypoints, descriptors


	def doExtractOrb(self, energy): 
		T1 = Time.now()

		#detect keypoints
		t2 = Time.now()

		# normalize input array to uint8 (orb cannot handle float32)
		energy_n = ConvertImageToUint8(energy)

		keypoints=[]
		history=[]
		while True:
			if not self.fastThreshold:
				self.fastThreshold = 20
			if not self.maxFeatures:
				self.maxFeatures = self.max_num_keypoints
			self.detector.setFastThreshold(self.fastThreshold)
			self.detector.setMaxFeatures(self.maxFeatures)
			keypoints=self.detector.detect(energy_n)
			N = len(keypoints)
			print("orb fastThreshold " , self.fastThreshold , " maxFeatures ", self.maxFeatures, " got " , N , " keypoints")

			# after 4 "zeros" assume the image is simply wrong
			history.append(N)
			if history[-4:]==[0]*4:
			   print("Failed to extract keypoints")
			   break

			if self.min_num_keypoints>0.001 and N < self.min_num_keypoints :
				self.fastThreshold -= 1
				continue

			if self.max_num_keypoints>0.001 and N > self.max_num_keypoints :
				self.fastThreshold += 1
				continue

			break

		keypoints = self.detector.detect(energy_n)
		
		print("orb got " , len(keypoints) , " keypoints")

		msec_detect = t2.elapsedMsec()

		t2 = Time.now()
		if self.anms>0 and len(keypoints)>self.anms:

			# important!
			# sort keypoints in DESC order by response 
			# remember since anms need them to be in order
			keypoints=sorted(keypoints,key=lambda A: A.response,reverse=True)

			responses=[keypoint.response for keypoint in keypoints]
			xs=[keypoint.pt[0] for keypoint in keypoints]
			ys=[keypoint.pt[1] for keypoint in keypoints]
	
			good_indices = KeyPoint.adaptiveNonMaximalSuppression(responses,xs,ys,self.anms)

			keypoints=[keypoints[it] for it in good_indices]

		msec_anms=t2.elapsedMsec()

		# compute descriptors
		t2 = Time.now()
		keypoints,descriptors=self.detector.compute(energy_n, keypoints)
		msec_compute = t2.elapsedMsec()

		print("Extracted",len(keypoints), " keypoints"," in " , T1.elapsedMsec() ," msec", "msec_detect", msec_detect, "msec_compute" , msec_compute, "msec_anms",msec_anms)
		if self.timing:
			return keypoints, descriptors, T1.elapsedMsec() / 1000
		else:
			return keypoints, descriptors

	def doExtractSift(self, energy): 
		T1 = Time.now()

		#detect keypoints
		t2 = Time.now()

		# normalize input array to uint8 (sift cannot handle float32)
		energy_n = ConvertImageToUint8(energy)
    
		keypoints = self.detector.detect(energy_n)
		
		print("sift got " , len(keypoints) , " keypoints")

		msec_detect = t2.elapsedMsec()

		t2 = Time.now()
		if self.anms>0 and len(keypoints)>self.anms:

			# important!
			# sort keypoints in DESC order by response 
			# remember since anms need them to be in order
			keypoints=sorted(keypoints,key=lambda A: A.response,reverse=True)

			responses=[keypoint.response for keypoint in keypoints]
			xs=[keypoint.pt[0] for keypoint in keypoints]
			ys=[keypoint.pt[1] for keypoint in keypoints]
	
			good_indices = KeyPoint.adaptiveNonMaximalSuppression(responses,xs,ys,self.anms)

			keypoints=[keypoints[it] for it in good_indices]

		msec_anms=t2.elapsedMsec()

		# compute descriptors
		t2 = Time.now()

		keypoints,descriptors=self.detector.compute(energy_n, keypoints)

		if descriptors is not None:
			descriptors = np.array(descriptors, dtype=numpy.uint8)

		msec_compute = t2.elapsedMsec()

		print("Extracted",len(keypoints), " keypoints"," in " , T1.elapsedMsec() ," msec", "msec_detect", msec_detect, "msec_compute" , msec_compute, "msec_anms",msec_anms)
		if self.timing:
			return keypoints, descriptors, T1.elapsedMsec() / 1000
		else:
			return keypoints, descriptors
