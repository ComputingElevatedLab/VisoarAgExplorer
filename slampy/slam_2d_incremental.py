import os.path

from math import isnan

from slampy.image_provider import *
from slampy.image_provider_generic import ImageProviderGeneric
from slampy.image_provider_lumenera import ImageProviderLumenera
from slampy.image_provider_micasense import ImageProviderRedEdge
from slampy.image_provider_sequoia import ImageProviderSequoia
from slampy.image_utils import *

execution_times = {}


def writeTimesToCsv():
    if len(execution_times.keys()) == 0:
        return

    filename = "times.csv"
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "a") as file:
        for key in execution_times.keys():
            times = execution_times[key]
            file.write(f"{key},")
            total = 0
            for t in times:
                total += t
            file.write(f"{total},{total / len(times)},{min(times)},{max(times)},")
            for i in range(len(times)):
                file.write(f"{times[i]},")
            file.write("\n")
        file.close()


# Perform 2D SLAM tasks incrementally
class Slam2DIncremental(Slam):

    def __init__(self, directory, provider_type, extractor, verbose):
        super(Slam2DIncremental, self).__init__()
        self.depth = None
        self.width = 0
        self.height = 0
        self.dtype = DType()
        self.calibration = Calibration()
        self.image_dir = directory
        self.cache_dir = os.path.join(directory, "VisusSlamFiles")
        TryRemoveFiles(f"{self.cache_dir}/~*")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.debug_mode = False
        self.energy_size = 1280
        self.min_num_keypoints = 3000
        self.max_num_keypoints = 6000
        self.anms = 1300
        self.max_reproj_error = 0.01
        self.ratio_check = 0.8
        self.calibration.bFixed = False
        self.ba_tolerance = 0.005
        self.images = []
        self.keyframes = []
        self.extractor = None
        self.extractor_method = extractor
        self.micasense_band = 0
        self.physic_box = None
        self.enable_svg = True
        self.enable_color_matching = False
        self.do_bundle_adjustment = True
        self.num_converted = 0
        self.distance_threshold = None
        self.verbose = verbose
        self.centers = {}

        # Initialize the provider
        if provider_type == "lumenera":
            self.provider = ImageProviderLumenera()
        elif provider_type == "rededge":
            self.provider = ImageProviderRedEdge()
        elif provider_type == "sequoia":
            self.provider = ImageProviderSequoia()
        else:
            self.provider = ImageProviderGeneric()
        self.provider.skip_value = 1
        self.provider.telemetry = None
        self.provider.plane = None
        self.provider.image_dir = self.image_dir
        self.provider.cache_dir = self.cache_dir
        self.provider.extractor_method = "akaze"

    def addImage(self, image):
        func_name = "addImage"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.provider.addImage([image])
        self.images.append(self.provider.images[-1])

        if len(self.provider.images) == 1:
            multi = self.provider.generateMultiImage(self.provider.images[-1])
            image_as_array = Array.fromNumPy(InterleaveChannels(multi), TargetDim=2)
            self.width = image_as_array.getWidth()
            self.height = image_as_array.getHeight()
            self.depth = image_as_array.getDepth()
            self.dtype = image_as_array.dtype
            # NOTE: from telemetry I'm just taking lat,lon,alt,yaw (not other stuff)
            if self.provider.telemetry:
                self.provider.loadTelemetry(self.provider.telemetry)
            if not self.provider.calibration:
                self.provider.calibration = self.provider.guessCalibration(multi)
            self.calibration = self.provider.calibration
            self.physic_box = None

        self.provider.loadMetadata()
        self.provider.loadSensorCfg()

        self.provider.findPanels()
        self.provider.loadLatLonAltFromMetadata()
        self.provider.loadYawFromMetadata()

        self.provider.interpolateGPS()
        self.provider.plane = self.provider.guessPlane()
        self.provider.setPlane(self.provider.plane)

        latest_image = self.provider.images[-1]
        latest_image.yaw += self.provider.yaw_offset
        while latest_image.yaw > +math.pi:
            latest_image.yaw -= 2 * math.pi
        while latest_image.yaw < -math.pi:
            latest_image.yaw += 2 * math.pi
        print(latest_image.filenames[0], "yaw_radians", latest_image.yaw, "yaw_degrees", math.degrees(latest_image.yaw))

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return self.provider.images[-1]

    def addCamera(self, image):
        func_name = "addCamera"
        start_time = None
        if self.verbose:
            start_time = time.time()

        camera = Camera()
        camera.id = len(self.cameras)
        camera.color = Color.random()
        for filename in image.filenames:
            camera.filenames.append(filename)
        super().addCamera(camera)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return self.cameras[-1]

    def createIdx(self, camera):
        func_name = "createIdx"
        start_time = None
        if self.verbose:
            start_time = time.time()

        camera.idx_filename = f"./idx/{camera.id:04d}.idx"
        field = Field("myfield", self.dtype)
        field.default_layout = "row_major"
        CreateIdx(url=os.path.join(self.cache_dir, camera.idx_filename),
                  dim=2,
                  filename_template=f"./{camera.id:04d}.bin",
                  blocksperfile=-1,
                  fields=[field],
                  dims=(self.width, self.height))

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def getDistanceThreshold(self):
        func_name = "getDistanceThreshold"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.distance_threshold = self.distance(self.cameras[0].getWorldCenter(), self.cameras[-1].getWorldCenter())

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def getTrivialBaIndices(self):
        func_name = "getTrivialBaIndices"
        start_time = time.time()

        ret_val = [len(self.cameras) - 2, len(self.cameras) - 1]

        stop_time = time.time()
        if func_name not in execution_times:
            execution_times[func_name] = []
        execution_times[func_name].append(stop_time - start_time)

        return ret_val

    def getIntersectionBaIndices(self, index, previous):
        func_name = "getIntersectionBasedBaMask"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if self.distance_threshold is None:
            self.getDistanceThreshold()

        indices = [index]
        camera = self.cameras[index]
        camera.bFixed = False
        for i, other_camera in enumerate(self.cameras):
            if i == index:
                continue

            distance = self.distance(self.centers[camera.id], self.centers[other_camera.id])
            if distance > self.distance_threshold or i in previous:
                other_camera.bFixed = True
                continue

            self.findMatches(camera, other_camera)
            other_camera.color = camera.color
            other_camera.bFixed = False
            indices.append(i)

        for i in indices[1:]:
            camera = self.cameras[i]
            for j in previous:
                other_camera = self.cameras[j]

                distance = self.distance(self.centers[camera.id], self.centers[other_camera.id])
                if distance > self.distance_threshold:
                    continue

                if Quad.intersection(camera.quad, other_camera.quad):
                    self.findMatches(camera, other_camera)
                    other_camera.bFixed = False

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return indices

    def bundle(self):
        func_name = "bundle"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.bundleAdjustment(self.ba_tolerance)
        self.refreshQuads()

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def showEnergy(self, camera, energy):
        func_name = "showEnergy"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if self.debug_mode:
            SaveImage(self.cache_dir + "/energy/%04d.tif" % (camera.id,), energy)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def distance(self, p1, p2):
        func_name = "distance"
        start_time = None
        if self.verbose:
            start_time = time.time()

        ret_val = (((p2.x-p1.x)**2)+((p2.y-p1.y)**2)+((p2.z-p1.z)**2))**(1/2)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return ret_val

    def guessLocalCamera(self):
        func_name = "guessLocalCamera"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = self.getQuadsBox()
        x1i = math.floor(box.p1[0])
        x2i = math.ceil(box.p2[0])
        y1i = math.floor(box.p1[1])
        y2i = math.ceil(box.p2[1])
        rect = (x1i, y1i, (x2i - x1i), (y2i - y1i))
        subdiv = cv2.Subdiv2D(rect)
        find_camera = dict()
        center = self.cameras[-1].quad.centroid()
        center = (numpy.float32(center.x), numpy.float32(center.y))
        if center in find_camera:
            print(f"The following cameras are in the same position: {find_camera[center].id} {self.cameras[-1].id}")
        else:
            find_camera[center] = self.cameras[-1]
            subdiv.insert(center)

        cells, centers = subdiv.getVoronoiFacetList([])
        assert (len(cells) == len(centers))

        # Find edges
        edges = dict()
        for i in range(len(cells)):
            cell = cells[i]
            center = (numpy.float32(centers[i][0]), numpy.float32(centers[i][1]))
            camera = find_camera[center]

            for j in range(len(cell)):
                pt0 = cell[(j + 0) % len(cell)]
                pt1 = cell[(j + 1) % len(cell)]
                k0 = (pt0[0], pt0[1], pt1[0], pt1[1])
                k1 = (pt1[0], pt1[1], pt0[0], pt0[1])

                if k0 not in edges:
                    edges[k0] = set()
                if k1 not in edges:
                    edges[k1] = set()

                edges[k0].add(camera)
                edges[k1].add(camera)

        for i in edges:
            adjacent = tuple(edges[i])
            for A in range(0, len(adjacent) - 1):
                for B in range(A + 1, len(adjacent)):
                    camera1 = adjacent[A]
                    camera2 = adjacent[B]
                    camera1.addLocalCamera(camera2)

        # insert prev and next
        n_cameras = len(self.cameras)
        for i in range(n_cameras):
            camera2 = self.cameras[i]

            # insert prev and next
            if (i - 1) >= 0:
                camera2.addLocalCamera(self.cameras[i - 1])

            if (i + 1) < n_cameras:
                camera2.addLocalCamera(self.cameras[i + 1])

        new_local_cameras = {}
        for cameraI in self.cameras:
            new_local_cameras[cameraI] = set()
            for cameraJ in cameraI.getAllLocalCameras():
                # Heuristic to say: do not take cameras on the same drone flight "row"
                prev = self.previousCamera(cameraJ)
                next = self.nextCamera(cameraJ)
                if prev != cameraI and next != cameraI:
                    if prev:
                        new_local_cameras[cameraI].add(prev)
                    if next:
                        new_local_cameras[cameraI].add(next)

        for cameraI in new_local_cameras:
            for cameraJ in new_local_cameras[cameraI]:
                cameraI.addLocalCamera(cameraJ)

        # Draw the image
        w = float(box.size()[0])
        h = float(box.size()[1])

        W = int(4096)
        H = int(h * (W / w))
        out = numpy.zeros((H, W, 3), dtype="uint8")
        out.fill(255)

        def toScreen(p):
            func_name2 = "toScreen"
            start_time2 = None
            if self.verbose:
                start_time2 = time.time()

            ret_val = [int(0 + (p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))]

            if self.verbose:
                stop_time2 = time.time()
                if func_name2 not in execution_times:
                    execution_times[func_name2] = []
                execution_times[func_name2].append(stop_time2 - start_time2)

            return ret_val

        for i in range(len(cells)):
            cell = cells[i]
            center = (numpy.float32(centers[i][0]), numpy.float32(centers[i][1]))
            camera2 = find_camera[center]
            center = toScreen(center)
            cell = numpy.array([toScreen(it) for it in cell], dtype=numpy.int32)
            cv2.fillConvexPoly(out, cell, [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            cv2.polylines(out, [cell], True, [0, 0, 0], 3)
            cv2.putText(out, str(camera2.id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        [0, 0, 0])

        # SaveImage(self.cache_dir + "/~local_cameras.png", out)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def guessInitialPose(self):
        func_name = "guessInitialPose"
        start_time = None
        if self.verbose:
            start_time = time.time()

        lat0, lon0 = self.images[0].lat, self.images[0].lon
        img = self.images[-1]
        lat, lon, alt = img.lat, img.lon, img.alt
        x, y = GPSUtils.gpsToLocalCartesian(lat, lon, lat0, lon0)
        world_center = Point3d(x, y, alt)
        q = Quaternion(Point3d(0, 0, 1), -img.yaw) * Quaternion(Point3d(1, 0, 0), math.pi)
        self.cameras[-1].pose = Pose(q, world_center).inverse()

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def saveMidx(self, num_processed):
        func_name = "saveMidx"
        start_time = None
        if self.verbose:
            start_time = time.time()

        print("Saving midx")
        lat0, lon0 = self.images[0].lat, self.images[0].lon

        logic_box = self.getQuadsBox()

        # instead of working in range -180,+180 -90,+90 (worldwise ref frame) I normalize the range in [0,1]*[0,1]
        # physic box is in the range [0,1]*[0,1]
        # logic_box is in pixel coordinates
        # NOTE: I can override the physic box by command line
        physic_box = self.physic_box
        if physic_box is not None:
            print("Using physic_box forced by the user", physic_box.toString())
        else:
            physic_box = BoxNd.invalid()
            for I, camera in enumerate(self.cameras):
                quad = self.computeWorldQuad(camera)
                for point in quad.points:
                    lat, lon = GPSUtils.localCartesianToGps(point.x, point.y, lat0, lon0)
                    alpha, beta = GPSUtils.gpsToUnit(lat, lon)
                    physic_box.addPoint(PointNd(Point2d(alpha, beta)))

        logic_centers = []
        for I, camera in enumerate(self.cameras):
            p = camera.getWorldCenter()
            lat, lon, alt = *GPSUtils.localCartesianToGps(p.x, p.y, lat0, lon0), p.z
            alpha, beta = GPSUtils.gpsToUnit(lat, lon)
            alpha = (alpha - physic_box.p1[0]) / float(physic_box.size()[0])
            beta = (beta - physic_box.p1[1]) / float(physic_box.size()[1])
            logic_x = logic_box.p1[0] + alpha * logic_box.size()[0]
            logic_y = logic_box.p1[1] + beta * logic_box.size()[1]
            logic_centers.append((logic_x, logic_y))

        lines = []

        # this is the midx
        lines.append("<dataset typename='IdxMultipleDataset' logic_box='%s %s %s %s' physic_box='%s %s %s %s'>" % (
            cstring(int(logic_box.p1[0])), cstring(int(logic_box.p2[0])), cstring(int(logic_box.p1[1])),
            cstring(int(logic_box.p2[1])),
            cstring10(physic_box.p1[0]), cstring10(physic_box.p2[0]), cstring10(physic_box.p1[1]),
            cstring10(physic_box.p2[1])))
        lines.append("")

        # dump some information about the slam
        lines.append("<slam width='%s' height='%s' dtype='%s' calibration='%s %s %s' />" % (
            cstring(self.width), cstring(self.height), self.dtype.toString(),
            cstring(self.calibration.f), cstring(self.calibration.cx), cstring(self.calibration.cy)))
        lines.append("")

        if (isinstance(self.provider, ImageProviderRedEdge)):
            # if we're using a micasense camera, create a field for each band
            for n in range(0, len(self.images[0].filenames)):
                lines.append(f"<field name='band{n}'><code>output=voronoi()[{n}]</code></field>")

        else:
            # this is the default field
            lines.append("<field name='blend'><code>output=voronoi()</code></field>")

        lines.append("")

        # how to go from logic_box (i.e. pixel) -> physic box ([0,1]*[0,1])
        lines.append("<translate x='%s' y='%s'>" % (cstring10(physic_box.p1[0]), cstring10(physic_box.p1[1])))
        lines.append("<scale     x='%s' y='%s'>" % (
        cstring10(physic_box.size()[0] / logic_box.size()[0]), cstring10(physic_box.size()[1] / logic_box.size()[1])))
        lines.append("<translate x='%s' y='%s'>" % (cstring10(-logic_box.p1[0]), cstring10(-logic_box.p1[1])))
        lines.append("")

        if self.enable_svg:
            W = int(1024)
            H = int(W * (logic_box.size()[1] / float(logic_box.size()[0])))

            lines.append("<svg width='%s' height='%s' viewBox='%s %s %s %s' >" % (
                cstring(W),
                cstring(H),
                cstring(int(logic_box.p1[0])),
                cstring(int(logic_box.p1[1])),
                cstring(int(logic_box.p2[0])),
                cstring(int(logic_box.p2[1]))))

            lines.append("<g stroke='#000000' stroke-width='1' fill='#ffff00' fill-opacity='0.3'>")
            for I, camera in enumerate(self.cameras):
                lines.append("\t<poi point='%s,%s' />" % (cstring(logic_centers[I][0]), cstring(logic_centers[I][1])))
            lines.append("</g>")

            lines.append("<g fill-opacity='0.0' stroke-opacity='0.5' stroke-width='2'>")
            for I, camera in enumerate(self.cameras):
                lines.append("\t<polygon points='%s' stroke='%s' />" % (
                camera.quad.toString(",", " "), camera.color.toString()[0:7]))
            lines.append("</g>")

            lines.append("</svg>")
            lines.append("")

        for I, camera in enumerate(self.cameras):
            p = camera.getWorldCenter()
            lat, lon, alt = *GPSUtils.localCartesianToGps(p.x, p.y, lat0, lon0), p.z
            lines.append(
                "<dataset url='%s' color='%s' quad='%s' filenames='%s' q='%s' t='%s' lat='%s' lon='%s' alt='%s' />" % (
                    camera.idx_filename,
                    camera.color.toString(),
                    camera.quad.toString(),
                    ";".join(camera.filenames),
                    camera.pose.q.toString(),
                    camera.pose.t.toString(),
                    cstring10(lat), cstring10(lon), cstring10(alt)))

        lines.append("")
        lines.append("</translate>")
        lines.append("</scale>")
        lines.append("</translate>")
        lines.append("")
        lines.append("</dataset>")

        SaveTextDocument(f"{self.cache_dir}/visus{num_processed}.midx", "\n".join(lines))
        print("Midx Saved")

        print("Saving google")
        SaveTextDocument(self.cache_dir + "/google.midx",
                         """
                         <dataset name='slam' typename='IdxMultipleDataset'>
                             <field name='voronoi'><code>output=voronoi()</code></field>
                             <dataset typename='GoogleMapsDataset' tiles='http://mt1.google.com/vt/lyrs=s' physic_box='0.0 1.0 0.0 1.0' />
                             <dataset name='visus'   url='./visus.midx' />
                         </dataset>
                         """)
        print("Google Saved")

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def debugMatchesGraph(self):
        func_name = "debugMatchesGraph"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = self.getQuadsBox()

        w = float(box.size()[0])
        h = float(box.size()[1])

        if isnan(w) or isnan(h):
            return

        W = int(4096)
        H = int(h * (W / w))
        out = numpy.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        def getImageCenter(image):
            func_name2 = "getImageCenter"
            start_time2 = None
            if self.verbose:
                start_time2 = time.time()

            p = image.quad.centroid()
            ret_val = int(0 + (p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))

            if self.verbose:
                stop_time2 = time.time()
                if func_name2 not in execution_times:
                    execution_times[func_name2] = []
                execution_times[func_name2].append(stop_time2 - start_time2)

            return ret_val

        for bGoodMatches in [False, True]:
            for cameraI in self.cameras:
                local_cameras = cameraI.getAllLocalCameras()
                for j in range(local_cameras.size()):
                    cameraJ = local_cameras[j]
                    edge = cameraI.getEdge(cameraJ)
                    if cameraI.id < cameraJ.id and bGoodMatches == (True if edge.isGood() else False):
                        p0 = getImageCenter(cameraI)
                        p1 = getImageCenter(cameraJ)
                        color = [0, 0, 0, 255] if edge.isGood() else [211, 211, 211, 255]
                        cv2.line(out, p0, p1, color, 1)
                        num_matches = edge.matches.size()
                        if num_matches > 0:
                            cx = int(0.5 * (p0[0] + p1[0]))
                            cy = int(0.5 * (p0[1] + p1[1]))
                            cv2.putText(out, str(num_matches), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color)

        for camera in self.cameras:
            center = getImageCenter(camera)
            cv2.putText(out, str(camera.id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        [0, 0, 0, 255])

        SaveImage(GuessUniqueFilename(self.cache_dir + "/~matches%d.png"), out)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def debugSolution(self):
        func_name = "debugSolution"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        out = numpy.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        def toScreen(p):
            func_name2 = "toScreen2"
            start_time2 = None
            if self.verbose:
                start_time2 = time.time()

            ret_val = int(0 + (p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))

            if self.verbose:
                stop_time2 = time.time()
                if func_name2 not in execution_times:
                    execution_times[func_name2] = []
                execution_times[func_name2].append(stop_time2 - start_time2)

            return ret_val

        for camera in self.cameras:
            color = (
            int(255 * camera.color.getRed()), int(255 * camera.color.getGreen()), int(255 * camera.color.getBlue()),
            255)
            points = numpy.array([toScreen(it) for it in camera.quad.points], dtype=numpy.int32)
            cv2.polylines(out, [points], True, color, 3)
            cv2.putText(out, str(camera.id), toScreen(camera.quad.points[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        color)

        SaveImage(GuessUniqueFilename(self.cache_dir + "/~solution%d.png"), out)

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def convertAndExtract(self):
        func_name = "convertAndExtract"
        start_time = None
        if self.verbose:
            start_time = time.time()

        extraction_start = time.time()
        img = self.images[-1]
        camera = self.cameras[-1]
        self.centers[camera.id] = camera.getWorldCenter()

        if not self.extractor:
            self.extractor = ExtractKeyPoints(self.min_num_keypoints, self.max_num_keypoints, self.anms, self.extractor_method)

        if self.enable_color_matching:
            color_matching_ref = None

        print(f"extraction time in ms: {(time.time() - extraction_start) * 1000}")

        # create idx and extract keypoints
        keypoint_path = f"{self.cache_dir}/keypoints/{camera.id}"
        idx_path = f"{self.cache_dir}/{camera.idx_filename}"

        if not self.debug_mode and self.loadKeyPoints(camera, keypoint_path) and os.path.isfile(idx_path) and os.path.isfile(idx_path.replace(".idx", ".bin")):
            print("Keypoints already stored and idx generated: ", img.filenames[0])
            print(f"Done {camera.filenames[0]}")

            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)
            return

        generate_start = time.time()
        full = self.generateImage(img)
        print(f"Generated in: {(time.time() - generate_start) * 1000} ms")

        # Match Histograms
        if self.enable_color_matching:
            if color_matching_ref:
                print("Doing color matching...")
                MatchHistogram(full, color_matching_ref)
            else:
                color_matching_ref = full

        dataset_start = time.time()
        data = LoadDataset(idx_path)
        # slow: first write then compress
        data.write(full)
        # dataset.compressDataset(["lz4"],Array.fromNumPy(full,TargetDim=2, bShareMem=True)) # write zipped full
        # fast: compress in-place
        # ,"jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE" ,"jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE","jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE"]
        # write zipped full
        # data.compressDataset(["lz4"], Array.fromNumPy(full, TargetDim=2, bShareMem=True))
        print(f"Dataset write time: {(time.time() - dataset_start) * 1000} ms")

        # convert_start = time.time()
        # if we're using micasense imagery, select the band specified by the user for extraction
        if isinstance(self.provider, ImageProviderRedEdge):
            print(f"Using band index {self.micasense_band} for extraction")
            energy = full[:, :, self.micasense_band]
        else:
            energy = ConvertImageToGrayScale(full)
        energy = ResizeImage(energy, self.energy_size)
        (keypoints, descriptors) = self.extractor.doExtract(energy)

        vs = self.width / float(energy.shape[1])
        if keypoints:
            camera.keypoints.reserve(len(keypoints))
            for keypoint in keypoints:
                kp = KeyPoint(vs * keypoint.pt[0], vs * keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id)
                camera.keypoints.push_back(kp)
            camera.descriptors = Array.fromNumPy(descriptors, TargetDim=2)

        self.saveKeyPoints(camera, keypoint_path)

        energy = cv2.cvtColor(energy, cv2.COLOR_GRAY2RGB)
        for keypoint in keypoints:
            cv2.drawMarker(energy, (int(keypoint.pt[0]), int(keypoint.pt[1])), (0, 255, 255), cv2.MARKER_CROSS, 5)
        energy = cv2.flip(energy, 0)
        energy = ConvertImageToUint8(energy)
        # convert_stop = time.time()
        # print(f"Convert time: {(convert_stop - convert_start) * 1000} ms")
        self.showEnergy(camera, energy)

        print(f"Done converting and extracting {camera.filenames[0]}")

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

    def findMatches(self, camera1, camera2):
        func_name = "findMatches"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if camera1.keypoints.empty() or camera2.keypoints.empty():
            camera2.getEdge(camera1).setMatches([], "No keypoints")
            return 0

        # We have already set matches for these two cameras
        if camera2.getEdge(camera1).isGood():
            return 0

        matches, H21, err = FindMatches(self.width, self.height,
                                        camera1.id, [(k.x, k.y) for k in camera1.keypoints],
                                        Array.toNumPy(camera1.descriptors),
                                        camera2.id, [(k.x, k.y) for k in camera2.keypoints],
                                        Array.toNumPy(camera2.descriptors),
                                        self.max_reproj_error * self.width, self.ratio_check)

        if self.debug_mode and H21 is not None and len(matches) > 0:
            points1 = [(k.x, k.y) for k in camera1.keypoints]
            points2 = [(k.x, k.y) for k in camera2.keypoints]

            a = Array.toNumPy(ArrayUtils.loadImage(f"{self.cache_dir}/energy/{camera1.id}.tif"))
            b = Array.toNumPy(ArrayUtils.loadImage(f"{self.cache_dir}/energy/{camera2.id}.tif"))
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

            DebugMatches(
                f"{self.cache_dir}/debug_matches/{err if err else 'good'}/{camera1.id}.{camera2.id}.{len(matches)}.png",
                self.width,
                self.height,
                a,
                [points1[match.queryIdx] for match in matches],
                H21,
                b,
                [points2[match.trainIdx] for match in matches],
                numpy.identity(3, dtype='float32')
                )

        if err:
            camera2.getEdge(camera1).setMatches([], err)
            return 0

        matches = [Match(match.queryIdx, match.trainIdx, match.imgIdx, match.distance) for match in matches]
        camera2.getEdge(camera1).setMatches(matches, str(len(matches)))

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return len(matches)

    def generateImage(self, img):
        func_name = "generateImage"
        start_time = None
        if self.verbose:
            start_time = time.time()

        print("Generating image", img.filenames[0])
        image = InterleaveChannels(self.provider.generateMultiImage(img))

        if self.verbose:
            stop_time = time.time()
            if func_name not in execution_times:
                execution_times[func_name] = []
            execution_times[func_name].append(stop_time - start_time)

        return image


# Compose a new image from two provided components on a given axis.
def ComposeImage(layers, axis):
    func_name = "composeImage"
    start_time = time.time()

    h = [warped.shape[0] for warped in layers]
    w = [energy.shape[1] for energy in layers]
    w, h = [(sum(w), max(h)), (max(w), sum(h))][axis]
    image = numpy.zeros(shape=[h, w], dtype=layers[0].dtype)
    current = [0, 0]
    for single in layers:
        h, w = single.shape[0], single.shape[1]
        image[current[1]:current[1] + h, current[0]:current[0] + w, :] = single
        current[axis] += [w, h][axis]

    stop_time = time.time()
    if func_name not in execution_times:
        execution_times[func_name] = []
    execution_times[func_name].append(stop_time - start_time)

    return image


def SaveDatasetPreview(db_filename, img_filename, width=1024):
    func_name = "saveDatasetPreview"
    start_time = time.time()

    db = LoadDataset(db_filename)
    maxh = db.getMaxResolution()
    logic_box = db.getLogicBox()
    logic_size = db.getLogicSize()
    print(f"Dataset has logic box {logic_box}, logic size{logic_size}")
    height = int(width * (logic_size[1] / logic_size[0]))
    deltah = int(math.log2((logic_size[0] / width) * (logic_size[1] / height)))
    data = db.read(logic_box=logic_box, max_resolution=(maxh - deltah))
    SaveImage(img_filename, data)

    stop_time = time.time()
    if func_name not in execution_times:
        execution_times[func_name] = []
    execution_times[func_name].append(stop_time - start_time)


class RedirectLog:
    def __init__(self, filename):
        super().__init__()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'w')
        self.__stdout__ = sys.stdout
        self.__stderr__ = sys.stderr
        self.__excepthook__ = sys.excepthook
        sys.stdout = self
        sys.stderr = self
        sys.excepthook = self.excepthook

    def excepthook(self, exctype, value, traceback):
        func_name = "exceptHook"
        start_time = time.time()

        sys.stdout = self.__stdout__
        sys.stderr = self.__stderr__
        sys.excepthook = self.__excepthook__
        sys.excepthook(exctype, value, traceback)

        stop_time = time.time()
        if func_name not in execution_times:
            execution_times[func_name] = []
        execution_times[func_name].append(stop_time - start_time)

    def write(self, msg):
        func_name = "write"
        start_time = time.time()

        msg = msg.replace(f"\n\n{datetime.datetime.now()[0:-7]} ")
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
        self.log.write(msg)

        stop_time = time.time()
        if func_name not in execution_times:
            execution_times[func_name] = []
        execution_times[func_name].append(stop_time - start_time)

    def flush(self):
        func_name = "flush"
        start_time = time.time()

        sys.__stdout__.flush()
        self.log.flush()

        stop_time = time.time()
        if func_name not in execution_times:
            execution_times[func_name] = []
        execution_times[func_name].append(stop_time - start_time)
