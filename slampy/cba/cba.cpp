#ifndef _CBA__
#define _CBA__

#include <cuda_bundle_adjustment.h>


#include <map>
#include <fstream>
#include <iterator>
#include <thread>
#include <numeric>
#include <vector>


#if WIN32
#include <direct.h>
#else
#include <unistd.h>
#endif



class CBA{
	public:
		cuba::CudaBundleAdjustment::Ptr optimizerGPU;
		cuba::CameraParams camera;
		Eigen::Matrix<double, 3, 1> Kvertex;
		cuba::LandmarkVertex *landMark;
		std::vector<cuba::PoseVertex*> poses;
		std::vector<cuba::StereoEdge*> edges;
		int calID;
		
		
	public:
	CBA(){
		optimizerGPU = cuba::CudaBundleAdjustment::create();
	}
	~CBA(){
		for(auto &pv: poses){
			delete pv;
		}
		for(auto &se: edges){
			delete se;
		}
	}
	
	void initialize(){
		optimizerGPU->initialize();
	}
	
	void optimize(){		
		optimizerGPU->optimize(1);
	}
	
	// calibration in rt-slam only has f, cx, cy
	void setCalibration(double &fx, double &fy, double &cx, double &cy, double &bf){		
		camera.fx = fx;
		camera.fy = fy;
		camera.cx = cx;
		camera.cy = cy;
		camera.bf = bf;
	}
	
	void setCalibrationVertex(int &id, double (&ary)[3], int &fixed){
		calID = id;
		Kvertex = Eigen::Matrix<double, 3, 1>(ary);
		*landMark = cuba::LandmarkVertex(id, Kvertex, fixed);
		optimizerGPU->addLandmarkVertex(landMark);
		
	}	
	
	void addVertex(int &id, double (&qin)[4], double (&tin)[3], int &fixed){
		auto q = Eigen::Quaterniond(qin);
		auto t = Eigen::Matrix<double, 3, 1>(tin);
		auto pv = new cuba::PoseVertex(id, q, t, camera, fixed);
		poses.push_back(pv);
		optimizerGPU->addPoseVertex(pv);
	}
	
	void addEdge(int &id1, int &id2, double (&ary)[4]){
		auto measurement = Eigen::Matrix<double, 4, 1>(ary); // cuba only accepts size 3
		auto info = Eigen::Matrix<double, 3, 1>::Identity(); // cuba expects a double, not array
		auto v1 = optimizerGPU->poseVertex(id1);
		auto v2 = optimizerGPU->poseVertex(calID); // cuba wants a LandmarkVertex, but we only have PoseVertex
		auto edge = new cuba::StereoEdge(measurement, info, v1, v2);
		edges.push_back(edge);
		optimizerGPU->addStereoEdge(edge);
	}
	
	void optimize(int n){
		optimizerGPU->optimize(n)
	}
	
	
	
	
};

int main(){
	//auto optimizerGPU = cuba::CudaBundleAdjustment::create();
	//CBA cba = CBA();
	printf("\n\nPass\n\n");
	
	return 0;
}

#endif

/*


*/