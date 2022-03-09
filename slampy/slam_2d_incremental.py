from slampy.image_provider import *
from slampy.image_provider_generic import ImageProviderGeneric
from slampy.image_provider_lumenera import ImageProviderLumenera
from slampy.image_provider_micasense import ImageProviderRedEdge
from slampy.image_provider_sequoia import ImageProviderSequoia
from slampy.image_utils import *

# Perform 2D SLAM tasks incrementally
class Slam2DIncremental(Slam):

    def __init__(self, directory, provider_type, extractor):
        super(Slam2DIncremental, self).__init__()
        self.width = 0
        self.height = 0
        self.dtype = DType()
        self.calibration = Calibration()
        self.image_dir = directory
        self.cache_dir = os.path.join(directory, "VisusSlamFiles")
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
        self.provider.calibration = None

    def addCamera(self, img):
        self.images.append(img)
        camera = Camera()
        camera.id = len(self.cameras)
        camera.color = Color.random()
        for filename in img.filenames:
            camera.filenames.append(filename)
        super().addCamera(camera)
        return camera

    def createIdx(self, camera):
        camera.idx_filename = "./idx/{:04d}.idx".format(camera.id)
        field = Field("myfield", self.dtype)
        field.default_layout = "row_major"
        CreateIdx(url=self.cache_dir + "/" + camera.idx_filename,
                  dim=2,
                  filename_template="./{:04d}.bin".format(camera.id),
                  blocksperfile=-1,
                  fields=[field],
                  dims=(self.width, self.height))

    def startAction(self, N, message):
        print("Starting action", N, message, "...")

    def advanceAction(self, I):
        # print("Advance action",I)
        pass

    def endAction(self):
        print("End action")

    def showEnergy(self, camera, energy):
        if self.debug_mode:
            SaveImage(self.cache_dir + "/energy/%04d.tif" % (camera.id,), energy)

    def guessLocalCameras(self):
        box = self.getQuadsBox()
        x1i = math.floor(box.p1[0])
        x2i = math.ceil(box.p2[0])
        y1i = math.floor(box.p1[1])
        y2i = math.ceil(box.p2[1])
        rect = (x1i, y1i, (x2i - x1i), (y2i - y1i))
        subdiv = cv2.Subdiv2D(rect)
        find_camera = dict()
        for camera in self.cameras:
            center = camera.quad.centroid()
            center = (numpy.float32(center.x), numpy.float32(center.y))
            if center in find_camera:
                # TODO: Figure out why we are getting here so many times
                print(f"The following cameras seems to be in the same position: {find_camera[center].id} {camera.id}")
            else:
                find_camera[center] = camera
                subdiv.insert(center)

        cells, centers = subdiv.getVoronoiFacetList([])
        assert(len(cells) == len(centers))

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

        for k in edges:
            adjacent = tuple(edges[k])
            for A in range(0, len(adjacent) - 1):
                for B in range(A + 1, len(adjacent)):
                    camera1 = adjacent[A]
                    camera2 = adjacent[B]
                    camera1.addLocalCamera(camera2)

        # insert prev and next
        n_cameras = self.cameras.size()
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
            return [int(0 + (p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))]

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

        SaveImage(self.cache_dir + "/~local_cameras.png", out)

    def guessInitialPoses(self):
        lat0, lon0 = self.images[0].lat, self.images[0].lon
        for i, camera in enumerate(self.cameras):
            lat, lon, alt = self.images[i].lat, self.images[i].lon, self.images[i].alt
            x, y = GPSUtils.gpsToLocalCartesian(lat, lon, lat0, lon0)
            world_center = Point3d(x, y, alt)
            img = self.images[i]
            q = Quaternion(Point3d(0, 0, 1), -img.yaw) * Quaternion(Point3d(1, 0, 0), math.pi)
            camera.pose = Pose(q, world_center).inverse()

    def saveMidx(self):
        print("Saving midx")
        lat0, lon0 = self.images[0].lat, self.images[0].lon
        logic_box = self.getQuadsBox()

        # instead of working in range -180,+180 -90,+90 (worldwise ref frame) I normalize the range in [0,1]*[0,1]
        # physic box is in the range [0,1]*[0,1]
        # logic_box is in pixel coordinates
        # NOTE: I can override the physic box by command line
        physic_box = self.physic_box
        if physic_box is not None:
            print("Using the user-provided physic_box", physic_box.toString())
        else:
            physic_box = BoxNd.invalid()
            for camera in self.cameras:
                quad = self.computeWorldQuad(camera)
                for point in quad.points:
                    lat, lon = GPSUtils.localCartesianToGps(point.x, point.y, lat0, lon0)
                    alpha, beta = GPSUtils.gpsToUnit(lat, lon)
                    physic_box.addPoint(PointNd(Point2d(alpha, beta)))

        logic_centers = []
        for camera in self.cameras:
            p = camera.getWorldCenter()
            lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, lat0, lon0)
            alpha, beta = GPSUtils.gpsToUnit(lat, lon)
            alpha = (alpha - physic_box.p1[0]) / float(physic_box.size()[0])
            beta = (beta - physic_box.p1[1]) / float(physic_box.size()[1])
            logic_x = logic_box.p1[0] + alpha * logic_box.size()[0]
            logic_y = logic_box.p1[1] + beta * logic_box.size()[1]
            logic_centers.append((logic_x, logic_y))

        lines = [
            f"<dataset typename='IdxMultipleDataset' logic_box='{int(logic_box.p1[0])} {int(logic_box.p2[0])} {int(logic_box.p1[1])} {int(logic_box.p2[1])}' physic_box='{physic_box.p1[0]} {physic_box.p2[0]} {physic_box.p1[1]} {physic_box.p2[1]}'>\n",
            f"<slam width='{self.width}' height='{self.height}' dtype='{self.dtype.toString()}' calibration='{self.calibration.f} {self.calibration.cx} {self.calibration.cy}' />\n",
        ]

        if isinstance(self.provider, ImageProviderRedEdge):
            # if we're using a micasense camera, create a field for each band
            for i in range(len(self.images[0].filenames)):
                lines.append(f"<field name='band{i}'><code>output=voronoi()[{i}]</code></field>")

        else:
            # this is the default field
            lines.append("<field name='blend'><code>output=voronoi()</code></field>")

        lines.append("")

        # how to go from logic_box (i.e. pixel) -> physic box ([0,1]*[0,1])
        lines.append(f"<translate x='{physic_box.p1[0]}' y='{physic_box.p1[1]}'>")
        lines.append(
            f"<scale     x='{physic_box.size()[0] / logic_box.size()[0]}' y='{physic_box.size()[1] / logic_box.size()[1]}'>")
        lines.append(f"<translate x='{-logic_box.p1[0]}' y='{-logic_box.p1[1]}'>")
        lines.append("")

        if self.enable_svg:
            w = int(1024)
            h = int(w * (logic_box.size()[1] / float(logic_box.size()[0])))

            lines.append(f"<svg width='{w}' height='{h}' viewBox='{int(logic_box.p1[0])} {int(logic_box.p1[1])} {int(logic_box.p2[0])} {int(logic_box.p2[1])}' >")
            lines.append("<g stroke='#000000' stroke-width='1' fill='#ffff00' fill-opacity='0.3'>")
            for i, camera in enumerate(self.cameras):
                lines.append(f"\t<poi point='{logic_centers[i][0]},{logic_centers[i][1]}'/>")
            lines.append("</g>")
            lines.append("<g fill-opacity='0.0' stroke-opacity='0.5' stroke-width='2'>")
            for camera in self.cameras:
                lines.append(
                    f"\t<polygon points='{camera.quad.toString(',', ' ')}' stroke='{camera.color.toString()[0:7]}' />")
            lines.append("</g>")
            lines.append("</svg>")
            lines.append("")

        for camera in self.cameras:
            p = camera.getWorldCenter()
            lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, lat0, lon0)
            alt = p.z
            lines.append(
                f"<dataset url='{camera.idx_filename}' color='{camera.color.toString()}' quad='{camera.quad.toString()}' filenames='{';'.join(camera.filenames)}' q='{camera.pose.q.toString()}' t='{camera.pose.t.toString()}' lat='{lat}' lon='{lon}' alt='{alt}' />")
        lines.append("")
        lines.append("</translate>")
        lines.append("</scale>")
        lines.append("</translate>")
        lines.append("")
        lines.append("</dataset>")

        SaveTextDocument(f"{self.cache_dir}/visus.midx", "\n".join(lines))
        print("Saved visus.midx")

        SaveTextDocument(self.cache_dir + "/google.midx",
                         """
<dataset name='slam' typename='IdxMultipleDataset'>
    <field name='voronoi'><code>output=voronoi()</code></field>
    <dataset typename='GoogleMapsDataset' tiles='http://mt1.google.com/vt/lyrs=s' physic_box='0.0 1.0 0.0 1.0' />
    <dataset name='visus'   url='./visus.midx' />
</dataset>
""")
        print("Saved google.midx")

    def debugMatchesGraph(self):

        box = self.getQuadsBox()

        w = float(box.size()[0])
        h = float(box.size()[1])

        W = int(4096)
        H = int(h * (W / w))
        out = numpy.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        def getImageCenter(camera):
            p = camera.quad.centroid()
            return (
                int(0 + (p[0] - box.p1[0]) * (W / w)),
                int(H - (p[1] - box.p1[1]) * (H / h)))

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

    def debugSolution(self):

        box = self.getQuadsBox()

        w = float(box.size()[0])
        h = float(box.size()[1])

        W = int(4096)
        H = int(h * (W / w))
        out = numpy.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        def toScreen(p):
            return (
                int(0 + (p[0] - box.p1[0]) * (W / w)),
                int(H - (p[1] - box.p1[1]) * (H / h)))

        for camera2 in self.cameras:
            color = (
                int(255 * camera2.color.getRed()), int(255 * camera2.color.getGreen()),
                int(255 * camera2.color.getBlue()),
                255)
            points = numpy.array([toScreen(it) for it in camera2.quad.points], dtype=numpy.int32)
            cv2.polylines(out, [points], True, color, 3)
            cv2.putText(out, str(camera2.id), toScreen(camera2.quad.points[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        color)

        SaveImage(GuessUniqueFilename(self.cache_dir + "/~solution%d.png"), out)

    def doPostIterationAction(self):
        self.debugSolution()
        self.debugMatchesGraph()

    def convertAndExtract(self, args):
        extraction_start = time.time()
        i, (img, camera) = args
        if not self.extractor:
            self.extractor = ExtractKeyPoints(self.min_num_keypoints, self.max_num_keypoints, self.anms, self.extractor_method)
        if self.enable_color_matching:
            color_matching_ref = None

        self.advanceAction(i)
        print(f"extraction time in ms: {(time.time() - extraction_start) * 1000}")

        # create idx and extract keypoints
        keypoint_filename = f"{self.cache_dir}/keypoints/{camera.id}"
        idx_filename = f"{self.cache_dir}/{camera.idx_filename}"

        if not self.debug_mode and self.loadKeyPoints(camera, keypoint_filename) and os.path.isfile(idx_filename) and os.path.isfile(idx_filename.replace(".idx", ".bin")):
            print("Keypoints already stored and idx generated: ", img.filenames[0])
        else:
            generate_start = time.time()
            full = self.generateImage(img)
            Assert(isinstance(full, numpy.ndarray))
            generate_stop = time.time()
            print(f"Generate time: {(generate_stop - generate_start) * 1000} ms")

            # Match Histograms
            if self.enable_color_matching:
                if color_matching_ref:
                    print("Doing color matching...")
                    MatchHistogram(full, color_matching_ref)
                else:
                    color_matching_ref = full

            dataset_start = time.time()
            data = LoadDataset(idx_filename)
            # slow: first write then compress
            # dataset.write(full)
            # dataset.compressDataset(["lz4"],Array.fromNumPy(full,TargetDim=2, bShareMem=True)) # write zipped full
            # fast: compress in-place
            # ,"jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE" ,"jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE","jpg-JPEG_QUALITYGOOD-JPEG_SUBSAMPLING_420-JPEG_OPTIMIZE"]
            # write zipped full
            data.compressDataset(["lz4"], Array.fromNumPy(full, TargetDim=2, bShareMem=True))
            dataset_stop = time.time()
            print(f"Dataset write time: {(dataset_stop - dataset_start) * 1000} ms")

            convert_start = time.time()
            energy = None
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
                camera.keypoints.clear()
                camera.keypoints.reserve(len(keypoints))
                for keypoint in keypoints:
                    camera.keypoints.push_back(KeyPoint(vs * keypoint.pt[0], vs * keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id))
                camera.descriptors = Array.fromNumPy(descriptors, TargetDim=2)

            self.saveKeyPoints(camera, keypoint_filename)

            energy = cv2.cvtColor(energy, cv2.COLOR_GRAY2RGB)
            for keypoint in keypoints:
                cv2.drawMarker(energy, (int(keypoint.pt[0]), int(keypoint.pt[1])), (0, 255, 255), cv2.MARKER_CROSS, 5)
            energy = cv2.flip(energy, 0)
            energy = ConvertImageToUint8(energy)
            convert_stop = time.time()
            print(f"Convert time: {(convert_stop - convert_start) * 1000} ms")

            # Unreachable code
            # if False:
            #     quad_box = camera.quad.getBoundingBox()
            #     VS = self.energy_size / max(quad_box.size()[0], quad_box.size()[1])
            #     T = Matrix.scale(2, VS) * camera.homography * Matrix.scale(2, vs)
            #     quad_box = Quad(T, Quad(energy.shape[1], energy.shape[0])).getBoundingBox()
            #     warped = cv2.warpPerspective(energy, MatrixToNumPy(Matrix.translate(-quad_box.p1) * T),
            #                                  (int(quad_box.size()[0]), int(quad_box.size()[1])))
            #     energy = ComposeImage([warped, energy], 1)

            self.showEnergy(camera, energy)

        print(f"Done {camera.filenames[0]} {i} of {len(self.cameras)}")

    def convertToIdxAndExtractKeyPoints(self):
        start = time.time()
        # convert to idx and find keypoints (don't use threads for IO ! it will slow down things)
        # NOTE I'm disabling write-locks
        self.startAction(len(self.cameras), "Converting idx and extracting keypoints...")
        for i, (img, camera) in enumerate(zip(self.images, self.cameras)):
            self.convertAndExtract([i, (img, camera)])
        stop = time.time()
        print(f"Conversion and feature extraction time: {(stop - start) * 1000} ms")

    def findMatches(self, camera1, camera2):
        if camera1.keypoints.empty() or camera2.keypoints.empty():
            camera2.getEdge(camera1).setMatches([], "No keypoints")
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

            DebugMatches(f"{self.cache_dir}/debug_matches/{err if err else 'good'}/{camera1.id}.{camera2.id}.{len(matches)}.png",
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
        return len(matches)

    def findAllMatches(self):
        start = time.time()
        jobs = []
        for camera2 in self.cameras:
            for camera1 in camera2.getAllLocalCameras():
                if camera1.id < camera2.id:
                    jobs.append(lambda pair=(camera1, camera2): self.findMatches(pair[0], pair[1]))
        self.startAction(len(jobs), "Finding all matches")

        if self.debug_mode:
            num_matches = 0
            for i, job in enumerate(jobs):
                num_matches += job()
                self.advanceAction(i)
        else:
            results = RunJobsInParallel(jobs, advance_callback=lambda ndone: self.advanceAction(ndone))
            num_matches = sum(results)
        stop = time.time()
        print(f"Found {num_matches} matches in: {(stop - start) * 1000} ms")

    def generateImage(self, img):
        start = time.time()
        print("Generating image", img.filenames[0])
        image = InterleaveChannels(self.provider.generateMultiImage(img))
        stop = time.time()
        print(f"Done: {img.id}, range {ComputeImageRange(image)}, shape {image.shape}, dtype {image.dtype} in {(stop - start) * 1000}")
        return image


# Compose a new image from two provided components on a given axis.
def ComposeImage(layers, axis):
    h = [warped.shape[0] for warped in layers]
    w = [energy.shape[1] for energy in layers]
    w, h = [(sum(w), max(h)), (max(w), sum(h))][axis]
    shape = list(layers[0].shape)
    shape[0], shape[1] = h, w
    image = numpy.zeros(shape=shape, dtype=layers[0].dtype)
    current = [0, 0]
    for single in layers:
        h, w = single.shape[0], single.shape[1]
        image[current[1]:current[1] + h, current[0]:current[0] + w, :] = single
        current[axis] += [w, h][axis]
    return image


def SaveDatasetPreview(db_filename, img_filename, width=1024):
    db = LoadDataset(db_filename)
    maxh = db.getMaxResolution()
    logic_box = db.getLogicBox()
    logic_size = db.getLogicSize()
    print(f"Dataset has logic box {logic_box}, logic size{logic_size}")
    height = int(width * (logic_size[1] / logic_size[0]))
    deltah = int(math.log2((logic_size[0] / width) * (logic_size[1] / height)))
    data = db.read(logic_box=logic_box, max_resolution=(maxh - deltah))
    SaveImage(img_filename, data)


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
        sys.stdout = self.__stdout__
        sys.stderr = self.__stderr__
        sys.excepthook = self.__excepthook__
        sys.excepthook(exctype, value, traceback)

    def write(self, msg):
        msg = msg.replace(f"\n\n{datetime.datetime.now()[0:-7]} ")
        sys.__stdout__.write(msg)
        sys.__stdout__.flush()
        self.log.write(msg)

    def flush(self):
        sys.__stdout__.flush()
        self.log.flush()