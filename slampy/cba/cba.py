import ctypes

lib = ctypes.cdll.LoadLibrary("./ext/slampy/slampy/cba/libcba.so")

class CBA(object):
    def __init__(self):
        lib.CBA_new.argtypes = []
        lib.CBA_new.restype = ctypes.c_void_p

        lib.CBA_cleanup.argtypes = [ctypes.c_void_p]
        lib.CBA_cleanup.restype = ctypes.c_void_p

        lib.CBA_initialize.argtypes = [ctypes.c_void_p]
        lib.CBA_initialize.restype = ctypes.c_void_p

        lib.CBA_setCalibration.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.CBA_setCalibration.restype = ctypes.c_void_p

        lib.CBA_addCamera.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        lib.CBA_addCamera.restype = ctypes.c_void_p

        lib.CBA_addLandmarkVertex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double * 3, ctypes.c_int]
        lib.CBA_addLandmarkVertex.restype = ctypes.c_void_p

        lib.CBA_addVertex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double * 4, ctypes.c_double * 3, ctypes.c_int, ctypes.c_int]
        lib.CBA_addVertex.restype = ctypes.c_void_p

        lib.CBA_addEdge.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double * 2]
        lib.CBA_addEdge.restype = ctypes.c_void_p

        lib.CBA_optimize.argtypes = [ctypes.c_void_p]
        lib.CBA_optimize.restype = ctypes.c_void_p

        lib.CBA_optimize_n.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.CBA_optimize_n.restype = ctypes.c_void_p

        lib.CBA_clear.argtypes = [ctypes.c_void_p]
        lib.CBA_clear.restype = ctypes.c_void_p

        lib.CBA_getLandMark.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.CBA_getLandMark.restype = ctypes.POINTER(ctypes.c_double)

        lib.CBA_getPoseVertex.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.CBA_getPoseVertex.restype = ctypes.POINTER(ctypes.c_double)

        self.obj = lib.CBA_new()
        pass

    def cleanup(self):
        lib.CBA_cleanup(self.obj)
        pass

    def initialize(self):
        lib.CBA_initialize(self.obj)
        pass

    def setCalibration(self, fx, fy, cx, cy):
        lib.CBA_setCalibration(self.obj, fx, fy, cx, cy)
        pass

    def addCamera(self, fx, fy, cx, cy, bf):
        lib.CBA_addCamera(self.obj, fx, fy, cx, cy, bf)
        pass

    def addLandmarkVertex(self, id, ary, fixed):
        ary = (ctypes.c_double * 3)(*ary)
        lib.CBA_addLandmarkVertex(self.obj, id, ary, fixed)
        pass

    def addVertex(self, id, qin, tin, fixed, cameraID):
        qin = (ctypes.c_double * 4)(*qin)
        tin = (ctypes.c_double * 3)(*tin)
        lib.CBA_addVertex(self.obj, id, qin, tin, fixed, cameraID)
        pass

    def addEdge(self, id1, id2, ary):
        ary = (ctypes.c_double * 2)(*ary)
        lib.CBA_addEdge(self.obj, id1, id2, ary)
        pass

    def optimize(self):
        lib.CBA_optimize(self.obj)
        pass

    def optimize_n(self, n):
        lib.CBA_optimize_n(self.obj, n)
        pass

    def clear(self):
        lib.CBA_clear(self.obj)
        pass

    def getLandMark(self, id):
        ary = lib.CBA_getLandMark(self.obj, id)
        return [ary[0], ary[1], ary[2]]

    def getPoseVertex(self, id):
        ary = lib.CBA_getPoseVertex(self.obj, id)
        return [ary[0], ary[1], ary[2], ary[3], ary[4], ary[5], ary[6]]

    def getPoseRotated(self, id):
        ary = lib.CBA_getPoseVertex(self.obj, id)
        return [ary[3], ary[0], ary[1], ary[2], ary[4], ary[5], ary[6]]
