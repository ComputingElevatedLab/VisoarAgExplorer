#ifndef _CBA__
#define _CBA__

#include <cuda_bundle_adjustment.h>


#include <map>
#include <fstream>
#include <iterator>
#include <thread>
#include <numeric>
#include <vector>
#include <iostream>


#if WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif



class CBA{
	private:
		cuba::CudaBundleAdjustment::Ptr optimizerGPU;
		cuba::CameraParams* camera;
		std::vector<cuba::CameraParams*> cameras;
		//Eigen::Matrix<double, 3, 1> Kvertex;
		std::vector<cuba::LandmarkVertex*> landMarks;
		std::vector<cuba::PoseVertex*> poses;
		std::vector<cuba::MonoEdge*> edges;

	public:
		double landMarkData[3];
		double poseData[7];


	CBA(){
		optimizerGPU = cuba::CudaBundleAdjustment::create();

/*
		const cuba::RobustKernelType robustKernelType = cuba::RobustKernelType::HUBER;
	    const double deltaMono = sqrt(5.991);
	    const double deltaStereo = sqrt(7.815);

        optimizerGPU->setRobustKernels(robustKernelType, deltaMono, cuba::EdgeType::MONOCULAR);
        optimizerGPU->setRobustKernels(robustKernelType, deltaStereo, cuba::EdgeType::STEREO);
**/
	}

	void cleanup(){
		for(auto pv: poses){
			delete pv;
		}
		for(auto se: edges){
			delete se;
		}
		for(auto lmv: landMarks){
			delete lmv;
		}
		for(auto cam: cameras){
		    delete cam;
		}
	}

	void initialize(){
		optimizerGPU->initialize();
	}


	// calibration in rt-slam only has f, cx, cy
	void setCalibration(double fx, double fy, double cx, double cy){
	    camera = new cuba::CameraParams();
		camera->fx = fx;
		camera->fy = fy;
		camera->cx = cx;
		camera->cy = cy;
		camera->bf = 1.0; // Stereo baseline times fx
		cameras.push_back(camera);
	}

	void addCamera(double fx, double fy, double cx, double cy, double bf){
	    auto cam = new cuba::CameraParams();
	    cam->fx = fx;
		cam->fy = fy;
		cam->cx = cx;
		cam->cy = cy;
		cam->bf = bf;
		cameras.push_back(cam);
	}

	void addLandmarkVertex(int id, double (ary)[3], int fixed){
		auto xyz = Eigen::Matrix<double, 3, 1>(ary);
		auto lmvertex = new cuba::LandmarkVertex(id, xyz, fixed);
		landMarks.push_back(lmvertex);
		optimizerGPU->addLandmarkVertex(lmvertex);
	}


	void addVertex(int id, double qin[4], double (tin)[3], int fixed, int cameraID){
		auto q = Eigen::Quaterniond(qin[0], qin[1], qin[2], qin[3]);
		auto t = Eigen::Matrix<double, 3, 1>(tin);
		auto pv = new cuba::PoseVertex(id, q, t, *cameras[cameraID], fixed);
		poses.push_back(pv);
		optimizerGPU->addPoseVertex(pv);
	}

	void addEdge(int id1, int id2, double (ary)[2]){

		auto measurement = Eigen::Matrix<double, 2, 1>(ary); // cuba only accepts size 2
		auto info = 1.0; //Eigen::Matrix<double, 4, 1>::Identity(); // cuba expects a double, not array // Information is different even in cubas examples between gpu and cpu
		auto v1 = optimizerGPU->poseVertex(id1);
		auto v2 = optimizerGPU->landmarkVertex(id2); // cuba wants a LandmarkVertex, but we only have PoseVertex
		auto edge = new cuba::MonoEdge(measurement, info, v1, v2);
		edges.push_back(edge);
		optimizerGPU->addMonocularEdge(edge);

	}

	void optimize(){
		optimizerGPU->optimize(1);
	}

	void optimize(int n){
		optimizerGPU->optimize(n);
	}

	void clear(){
		optimizerGPU->clear();
	}

	double * getLandMark(int id){
		auto lmd = optimizerGPU->landmarkVertex(id)->Xw.data();
		landMarkData[0] = lmd[0];
		landMarkData[1] = lmd[1];
		landMarkData[2] = lmd[2];
		return landMarkData;
	}

	double * getPoseVertex(int id){
		auto pv = optimizerGPU->poseVertex(id);
		poseData[0] = pv->q.w();
		poseData[1] = pv->q.x();
		poseData[2] = pv->q.y();
		poseData[3] = pv->q.z();
		poseData[4] = pv->t.x();
		poseData[5] = pv->t.y();
		poseData[6] = pv->t.z();

		return poseData;
	}

};

extern "C"
{
	CBA * CBA_new(){return new CBA();}

	void CBA_cleanup(CBA * cba){cba->cleanup();}

	void CBA_initialize(CBA * cba){cba->initialize();}

	void CBA_setCalibration(CBA * cba, double fx, double fy, double cx, double cy){
		cba->setCalibration(fx,fy,cx,cy);
	}

	void CBA_addCamera(CBA * cba, double fx, double fy, double cx, double cy, double bf){
	    cba->addCamera(fx, fy, cx, cy, bf);
	}

	void CBA_addLandmarkVertex(CBA * cba, int id, double (ary)[3], int fixed){
		cba->addLandmarkVertex(id, ary, fixed);
	}


	void CBA_addVertex(CBA * cba, int id, double (qin)[4], double (tin)[3], int fixed, int cameraID){
		cba->addVertex(id,qin,tin,fixed, cameraID);
	}

	void CBA_addEdge(CBA * cba, int id1, int id2, double (ary)[2]){
		cba->addEdge(id1, id2, ary);
	}

	void CBA_optimize(CBA * cba){
		cba->optimize();
	}

	void CBA_optimize_n(CBA * cba, int n){
		cba->optimize(n);
	}

	void CBA_clear(CBA * cba){
		cba->clear();
	}

	double * CBA_getLandMark(CBA * cba, int id){
		return cba->getLandMark(id);
	}

	double * CBA_getPoseVertex(CBA * cba, int id){
		return cba->getPoseVertex(id);
	}

}


#endif
