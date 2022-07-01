import math
import os.path
import random
import time

import cv2
import numpy as np

import micasense.capture
import micasense.image
import micasense.panel
import OpenVisus as Visus
import slampy.image_provider as img_utils

from rtree import index
from slampy.extract_keypoints import ExtractKeyPoints
from slampy.find_matches import FindMatches, DebugMatches
from slampy.gps_utils import GPSUtils
from slampy.image_provider_generic import ImageProviderGeneric
from slampy.image_provider_lumenera import ImageProviderLumenera
from slampy.image_provider_micasense import ImageProviderRedEdge
from slampy.image_provider_sequoia import ImageProviderSequoia
from slampy.metadata_reader import MetadataReader


class Slam2DIncremental(Visus.Slam):

    def __init__(self, alt_threshold, directory, provider_type, extractor, extraction_band, verbose):
        super(Slam2DIncremental, self).__init__()
        self.depth = None
        self.width = 0
        self.height = 0
        self.dtype = Visus.DType()
        self.calibration = Visus.Calibration()
        self.image_dir = directory
        self.cache_dir = os.path.join(directory, "VisusSlamFiles")
        Visus.TryRemoveFiles(f"{self.cache_dir}/~*")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.debug_mode = False
        self.energy_size = None
        self.min_key_points = 1800
        self.max_key_points = 5000
        self.anms = 200
        self.max_reprojection_error = 0.01
        self.ratio_check = 0.8
        self.calibration.bFixed = False
        self.ba_tolerance = 0.00001
        self.extractor = ExtractKeyPoints(self.min_key_points, self.max_key_points, self.anms, extractor)
        self.physic_box = None
        self.enable_svg = True
        self.enable_color_matching = False
        self.do_bundle_adjustment = True

        self.execution_times = {}
        self.extraction_band = extraction_band
        self.verbose = verbose
        self.world_centers = {}
        self.initial_multi_image = None
        self.lat0 = None
        self.lon0 = None
        self.plane = None
        self.vs = None
        self.distance_threshold = np.inf
        self.previous_yaw = None
        self.reader = MetadataReader()
        self.idx = index.Index()
        self.coordinates = {}

        # Initialize the image provider
        self.alt_threshold = alt_threshold
        self.multi_band = False
        self.panels_found = False
        self.initialized = False
        if provider_type == "lumenera":
            self.provider = ImageProviderLumenera()
        elif provider_type == "rededge" or provider_type == "micasense":
            self.multi_band = True
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

    def add_image(self, image_path):
        func_name = "add_image"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.provider.addImage([image_path])
        image = self.provider.images[-1]
        self.load_image_metadata(image)

        if len(self.provider.images) == 1:
            self.initialize_slam(image)

        image.alt -= self.plane
        self.adjust_image_yaw(image)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return image

    def add_multi_band_image(self, image_path):
        func_name = "add_multi_band_image"
        start_time = None
        if self.verbose:
            start_time = time.time()

        band = int(image_path[-5]) - 1
        image_bands = []
        for i in range(5):
            if band == i:
                image_bands.append(image_path)
            else:
                ith_band_path = image_path[:-5] + str(i + 1) + image_path[-4:]
                image_bands.append(ith_band_path)
        self.provider.addImage(image_bands)

        if not self.panels_found:
            self.load_metadata(self.provider.images[-1])
            panel = micasense.panel.Panel(micasense.image.Image(image_bands[0]))
            if panel.panel_detected():
                return None

            if len(self.provider.images) == 1:
                self.provider.images.pop(0)
                return None

            self.provider.findPanels()
            self.panels_found = True

        if not self.provider.images:
            return None

        image = self.provider.images[-1]
        self.load_image_metadata(image)

        if image.alt < self.alt_threshold:
            print(f"Dropping image: altitude of {image.alt} is below the threshold")
            self.provider.images.pop()
            return None

        if not self.initialized:
            self.initialize_slam(image)
            self.initialized = True

        image.alt -= self.plane
        self.adjust_image_yaw(image)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return image

    def load_image_metadata(self, image):
        func_name = "load_image_metadata"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.load_metadata(image)
        self.load_gps_from_metadata(image)
        self.load_yaw_from_metadata(image)
        self.interpolate_gps()

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def load_metadata(self, image):
        func_name = "load_metadata"
        start_time = None
        if self.verbose:
            start_time = time.time()

        filename = image.filenames[self.extraction_band]
        image.metadata = self.reader.readMetadata(filename)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def load_gps_from_metadata(self, image):
        func_name = "load_gcs_from_metadata"
        start_time = None
        if self.verbose:
            start_time = time.time()

        print("Loading GPS latitude, longitude, and altitude from metadata")

        lat = img_utils.FindMetadata(image.metadata, ["GPSLatitude", "Latitude", "lat"])
        if not lat:
            raise Exception("Error: missing latitude from metadata")

        lon = img_utils.FindMetadata(image.metadata, ["GPSLongitude", "Longitude", "lon"])
        if not lon:
            raise Exception("Error: missing longitude from metadata")

        alt = img_utils.FindMetadata(image.metadata, ["GPSAltitude", "Altitude", "alt"])
        if not alt:
            raise Exception("Error: missing altitude from metadata")

        image.lat = float(image.metadata[lat])
        image.lon = float(image.metadata[lon])
        image.alt = float(image.metadata[alt])

        print(f"lat = {image.lat}, lon = {image.lon}, alt = {image.alt}")

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def load_yaw_from_metadata(self, image):
        func_name = "load_yaw_from_metadata"
        start_time = None
        if self.verbose:
            start_time = time.time()

        print(f"Loading yaw from metadata")

        yaw = img_utils.FindMetadata(image.metadata, ["Yaw", "GimbalYaw", "GimbalYawDegree", "yaw", "yaw(gps)"])

        if yaw in image.metadata:
            image.yaw = float(image.metadata[yaw])
            print(f"yaw = {image.yaw}")
        else:
            image.yaw = self.previous_yaw
            print(f"Did not find a yaw value, using the previous yaw={self.previous_yaw}")

        # Convert the yaw to radians if necessary
        if yaw and "radians" not in yaw.lower():
            if "degrees" in yaw.lower():
                image.yaw = np.radians(image.yaw)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def interpolate_gps(self):
        func_name = "interpolate_gps"
        start_time = None
        if self.verbose:
            start_time = time.time()

        n = len(self.provider.images)
        for i, image in enumerate(self.provider.images):
            if not self.wrong_gps(image):
                continue

            a = i - 1
            while a >= 0 and self.wrong_gps(self.provider.images[a]):
                a -= 1

            b = i + 1
            while b < n and self.wrong_gps(self.provider.images[b]):
                b += 1

            if 0 <= a < b < n:
                alpha = (i - a) / float(b - a)
                image.lat = (1 - alpha) * self.provider.images[a].lat + alpha * self.provider.images[b].lat
                image.lon = (1 - alpha) * self.provider.images[a].lon + alpha * self.provider.images[b].lon
                image.alt = (1 - alpha) * self.provider.images[a].alt + alpha * self.provider.images[b].alt
                print(f"Interpolating GPS: lat = {image.lat}, lon = {image.lon}, alt = {image.alt}")
            else:
                raise Exception(f"Error: could not interpolate GPS for {image.filenames[0]}")

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def wrong_gps(self, image):
        return image.lat == 0.0 or image.lon == 0.0 or image.alt == 0.0

    def initialize_slam(self, image):
        func_name = "initialize_slam"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if not image:
            return

        self.lat0 = image.lat
        self.lon0 = image.lon
        self.plane = self.guess_plane(image)

        self.provider.loadSensorCfg()

        self.initial_multi_image = self.generate_multi_image(image)
        interleaved_image = self.interleave_channels(self.initial_multi_image)
        image_as_array = Visus.Array.fromNumPy(interleaved_image, TargetDim=2)

        self.width = image_as_array.getWidth()
        self.height = image_as_array.getHeight()
        self.depth = image_as_array.getDepth()
        self.dtype = image_as_array.dtype

        # Used to resize all incoming images
        scale = 1 / 5
        energy_width = int(self.width * scale)
        energy_height = int(self.height * scale)
        self.energy_size = (energy_width, energy_height)

        # NOTE: from telemetry I'm just taking lat,lon,alt,yaw (not other stuff)
        if self.provider.telemetry:
            self.provider.loadTelemetry(self.provider.telemetry)
        if not self.provider.calibration:
            self.provider.calibration = self.provider.guessCalibration(self.initial_multi_image)

        self.calibration = self.provider.calibration

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def generate_multi_image(self, image):
        func_name = "generate_multi_image"
        start_time = None
        if self.verbose:
            start_time = time.time()

        multi_image = self.provider.generateMultiImage(image)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return multi_image

    def interleave_channels(self, multi_image):
        func_name = "interleave_channels"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if len(multi_image) == 1:
            return multi_image[0]

        interleaved_image = np.dstack(multi_image)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return interleaved_image

    def guess_plane(self, image):
        if self.multi_band:
            print(f"Guessing plane from micasense panels")
            alt = img_utils.FindMetadata(image.metadata, ["GPSAltitude"])
            if alt:
                return np.min([float(panel.metadata[alt]) for panel in self.provider.panels])

        abs = img_utils.FindMetadata(image.metadata, ["AbsoluteAltitude", "GPSAltitude"])
        rel = img_utils.FindMetadata(image.metadata, ["RelativeAltitude", "GPSAltitudeRef"])

        if abs and rel:
            return float(image.metadata[abs]) - float(image.metadata[rel])

        return 0

    def adjust_image_yaw(self, image):
        func_name = "adjust_image_yaw"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if self.multi_band:
            image.yaw -= self.provider.yaw_offset / 2
        else:
            image.yaw += self.provider.yaw_offset
            offset = 2 * np.pi
            while image.yaw > np.pi:
                image.yaw -= offset
            while image.yaw < -np.pi:
                image.yaw += offset
        print(f"{image.filenames[0]} - {image.yaw} radians {np.degrees(image.yaw)} degrees")

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def add_camera(self, image):
        func_name = "add_camera"
        start_time = None
        if self.verbose:
            start_time = time.time()

        camera = Visus.Camera()
        camera.id = len(self.cameras)
        camera.color = Visus.Color.random()
        for filename in image.filenames:
            camera.filenames.append(filename)
        super().addCamera(camera)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return self.cameras[-1]

    def create_idx(self, camera):
        func_name = "create_idx"
        start_time = None
        if self.verbose:
            start_time = time.time()

        camera.idx_filename = f"./idx/{camera.id:04d}.idx"
        field = Visus.Field("myfield", self.dtype)
        field.default_layout = "row_major"
        Visus.CreateIdx(url=os.path.join(self.cache_dir, camera.idx_filename),
                        dim=2,
                        filename_template=f"./{camera.id:04d}.bin",
                        blocksperfile=-1,
                        fields=[field],
                        dims=(self.width, self.height))

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def set_initial_pose(self, image, camera):
        func_name = "set_initial_pose"
        start_time = None
        if self.verbose:
            start_time = time.time()

        x, y = GPSUtils.gpsToLocalCartesian(image.lat, image.lon, self.lat0, self.lon0)
        self.world_centers[camera.id] = Visus.Point3d(x, y, image.alt)
        q = Visus.Quaternion(Visus.Point3d(0, 0, 1), -image.yaw) * Visus.Quaternion(Visus.Point3d(1, 0, 0), np.pi)
        camera.pose = Visus.Pose(q, self.world_centers[camera.id]).inverse()

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def refresh_quads(self):
        func_name = "refresh_quads"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.refreshQuads()

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def set_local_cameras(self):
        func_name = "set_local_cameras"
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
        center = (np.float32(center.x), np.float32(center.y))
        if center in find_camera:
            print(f"The following cameras are in the same position: {find_camera[center].id} {self.cameras[-1].id}")
        else:
            find_camera[center] = self.cameras[-1]
            subdiv.insert(center)

        cells, centers = subdiv.getVoronoiFacetList([])

        # Find edges
        edges = {}
        for i, cell in enumerate(cells):
            center = (np.float32(centers[i][0]), np.float32(centers[i][1]))
            camera = find_camera[center]
            for j, _ in enumerate(cell):
                pt0 = cell[j % len(cell)]
                pt1 = cell[(j + 1) % len(cell)]
                k0 = (pt0[0], pt0[1], pt1[0], pt1[1])
                k1 = (pt1[0], pt1[1], pt0[0], pt0[1])

                if k0 not in edges:
                    edges[k0] = set()
                if k1 not in edges:
                    edges[k1] = set()

                edges[k0].add(camera)
                edges[k1].add(camera)

        for edge in edges:
            adjacent = list(edges[edge])
            for i, camera1 in enumerate(adjacent):
                for camera2 in adjacent[i + 1:]:
                    camera1.addLocalCamera(camera2)

        # insert prev and next
        for i, camera in enumerate(self.cameras):
            # insert prev and next
            if i >= 1:
                camera.addLocalCamera(self.cameras[i - 1])

            if (i + 1) < len(self.cameras):
                camera.addLocalCamera(self.cameras[i + 1])

        new_local_cameras = {}
        for camera1 in self.cameras:
            new_local_cameras[camera1] = set()
            for camera2 in camera1.getAllLocalCameras():
                # Heuristic to say: do not take cameras on the same drone flight "row"
                prev_camera = self.previousCamera(camera2)
                next_camera = self.nextCamera(camera2)
                if prev_camera != camera1 and next_camera != camera2:
                    if prev_camera:
                        new_local_cameras[camera1].add(prev_camera)
                    if next_camera:
                        new_local_cameras[camera1].add(next_camera)

        for camera1 in new_local_cameras:
            for camera2 in new_local_cameras[camera1]:
                camera1.addLocalCamera(camera2)

        # Draw the image
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        out = np.zeros((H, W, 3), dtype="uint8")
        out.fill(255)

        for i, cell in enumerate(cells):
            center = (np.float32(centers[i][0]), np.float32(centers[i][1]))
            camera2 = find_camera[center]
            center = self.quad_to_screen(center, box)
            cell = np.array([self.quad_to_screen(it, box) for it in cell], dtype=np.int32)
            cv2.fillConvexPoly(out, cell, [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            cv2.polylines(out, [cell], True, [0, 0, 0], 3)
            cv2.putText(out, str(camera2.id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, [0, 0, 0])

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def get_distance_threshold(self, x, y):
        func_name = "get_distance_threshold"
        start_time = None
        if self.verbose:
            start_time = time.time()

        x_center = self.cameras[x].getWorldCenter()
        y_center = self.cameras[y].getWorldCenter()
        self.distance_threshold = self.cartesian_distance(x_center, y_center) * 1.5

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def insert_camera_into_spatial_index(self, camera):
        func_name = "insert_camera_into_spatial_index"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = camera.quad.getBoundingBox()
        self.coordinates[camera.id] = (box.p1[0], box.p1[1], box.p2[0], box.p2[1])
        self.idx.insert(camera.id, self.coordinates[camera.id])

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def get_intersecting_indices(self, at):
        func_name = "get_intersecting_indices"
        start_time = None
        if self.verbose:
            start_time = time.time()

        camera = self.cameras[at]
        camera.bFixed = False
        indices = list(self.idx.intersection(self.coordinates[camera.id]))
        for i, other_camera in enumerate(self.cameras):
            if i == camera.id:
                continue
            elif i in indices:
                self.find_matches(camera, other_camera)
                other_camera.color = camera.color
                other_camera.bFixed = False
            else:
                other_camera.bFixed = True

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return indices

    def get_local_ba_indices(self, index, previous):
        func_name = "get_local_ba_indices"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if previous:
            self.get_distance_threshold(index, previous[0])
        else:
            self.get_distance_threshold(0, -1)

        indices = [index]
        camera = self.cameras[index]
        camera.bFixed = False
        for i, other_camera in enumerate(self.cameras):
            if i == index:
                continue

            distance = self.cartesian_distance(self.world_centers[camera.id], self.world_centers[other_camera.id])
            if distance > self.distance_threshold or i in previous:
                other_camera.bFixed = True
                continue

            self.find_matches(camera, other_camera)
            other_camera.color = camera.color
            other_camera.bFixed = False
            indices.append(i)

        for i in indices[1:]:
            camera = self.cameras[i]
            for j in previous:
                other_camera = self.cameras[j]

                distance = self.cartesian_distance(self.world_centers[camera.id], self.world_centers[other_camera.id])
                if distance > self.distance_threshold:
                    continue

                if Visus.Quad.intersection(camera.quad, other_camera.quad):
                    self.find_matches(camera, other_camera)
                    other_camera.bFixed = False

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return indices

    # Cartesian distance between two (x, y, z) coordinates
    @staticmethod
    def cartesian_distance(p1, p2):
        return (((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2) + ((p2.z - p1.z) ** 2)) ** 0.5

    def bundle_adjust(self):
        func_name = "bundle_adjust"
        start_time = None
        if self.verbose:
            start_time = time.time()

        self.bundleAdjustment(self.ba_tolerance)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def save_midx(self, suffix=""):
        func_name = "save_midx"
        start_time = None
        if self.verbose:
            start_time = time.time()

        print("Saving midx")

        logic_box = self.getQuadsBox()

        # instead of working in range -180,+180 -90,+90 (worldwise ref frame) I normalize the range in [0,1]*[0,1]
        # physic box is in the range [0,1]*[0,1]
        # logic_box is in pixel coordinates
        # NOTE: I can override the physic box by command line
        physic_box = self.physic_box
        if physic_box is not None:
            print("Using physic_box forced by the user", physic_box.toString())
        else:
            physic_box = Visus.BoxNd.invalid()
            for camera in self.cameras:
                quad = self.computeWorldQuad(camera)
                for point in quad.points:
                    lat, lon = GPSUtils.localCartesianToGps(point.x, point.y, self.lat0, self.lon0)
                    alpha, beta = GPSUtils.gpsToUnit(lat, lon)
                    physic_box.addPoint(Visus.PointNd(Visus.Point2d(alpha, beta)))

        logic_centers = []
        for camera in self.cameras:
            p = camera.getWorldCenter()
            lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, self.lat0, self.lon0)
            alpha, beta = GPSUtils.gpsToUnit(lat, lon)
            alpha = (alpha - physic_box.p1[0]) / float(physic_box.size()[0])
            beta = (beta - physic_box.p1[1]) / float(physic_box.size()[1])
            logic_x = logic_box.p1[0] + alpha * logic_box.size()[0]
            logic_y = logic_box.p1[1] + beta * logic_box.size()[1]
            logic_centers.append((logic_x, logic_y))

        # This is the midx, dump some information about the slam
        lines = ["<dataset typename='IdxMultipleDataset' logic_box='%s %s %s %s' physic_box='%s %s %s %s'>" % (
            Visus.cstring(int(logic_box.p1[0])), Visus.cstring(int(logic_box.p2[0])),
            Visus.cstring(int(logic_box.p1[1])),
            Visus.cstring(int(logic_box.p2[1])),
            Visus.cstring10(physic_box.p1[0]), Visus.cstring10(physic_box.p2[0]), Visus.cstring10(physic_box.p1[1]),
            Visus.cstring10(physic_box.p2[1])), "",
                 "<slam width='%s' height='%s' dtype='%s' calibration='%s %s %s' />" % (
                     Visus.cstring(self.width), Visus.cstring(self.height), self.dtype.toString(),
                     Visus.cstring(self.calibration.f), Visus.cstring(self.calibration.cx),
                     Visus.cstring(self.calibration.cy)), ""]

        # If we're using a micasense camera, create a field for each band
        if self.multi_band:
            for i, _ in enumerate(self.provider.images[0].filenames):
                lines.append(f"<field name='band{i}'><code>output=ArrayUtils.split(voronoi())[{i}]</code></field>")
        else:
            lines.append("<field name='blend'><code>output=voronoi()</code></field>")

        lines.append("")

        # how to go from logic_box (i.e. pixel) -> physic box ([0,1]*[0,1])
        lines.append("<translate x='%s' y='%s'>" % (Visus.cstring10(physic_box.p1[0]), Visus.cstring10(physic_box.p1[1])))
        lines.append("<scale     x='%s' y='%s'>" % (
            Visus.cstring10(physic_box.size()[0] / logic_box.size()[0]),
            Visus.cstring10(physic_box.size()[1] / logic_box.size()[1])))
        lines.append("<translate x='%s' y='%s'>" % (Visus.cstring10(-logic_box.p1[0]), Visus.cstring10(-logic_box.p1[1])))
        lines.append("")

        if self.enable_svg:
            W = int(1024)
            H = int(W * (logic_box.size()[1] / float(logic_box.size()[0])))

            lines.append("<svg width='%s' height='%s' viewBox='%s %s %s %s' >" % (
                Visus.cstring(W),
                Visus.cstring(H),
                Visus.cstring(int(logic_box.p1[0])),
                Visus.cstring(int(logic_box.p1[1])),
                Visus.cstring(int(logic_box.p2[0])),
                Visus.cstring(int(logic_box.p2[1]))))

            lines.append("<g stroke='#000000' stroke-width='1' fill='#ffff00' fill-opacity='0.3'>")
            for i, camera in enumerate(self.cameras):
                lines.append("\t<poi point='%s,%s' />" % (Visus.cstring(logic_centers[i][0]), Visus.cstring(logic_centers[i][1])))
            lines.append("</g>")

            lines.append("<g fill-opacity='0.0' stroke-opacity='0.5' stroke-width='2'>")
            for camera in self.cameras:
                lines.append("\t<polygon points='%s' stroke='%s' />" % (
                    camera.quad.toString(",", " "), camera.color.toString()[0:7]))
            lines.append("</g>")

            lines.append("</svg>")
            lines.append("")

        for camera in self.cameras:
            p = camera.getWorldCenter()
            lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, self.lat0, self.lon0)
            lines.append(
                "<dataset url='%s' color='%s' quad='%s' filenames='%s' q='%s' t='%s' lat='%s' lon='%s' alt='%s' />" % (
                    camera.idx_filename,
                    camera.color.toString(),
                    camera.quad.toString(),
                    ";".join(camera.filenames),
                    camera.pose.q.toString(),
                    camera.pose.t.toString(),
                    Visus.cstring10(lat), Visus.cstring10(lon), Visus.cstring10(p.z)))

        lines.append("")
        lines.append("</translate>")
        lines.append("</scale>")
        lines.append("</translate>")
        lines.append("")
        lines.append("</dataset>")

        Visus.SaveTextDocument(f"{self.cache_dir}/visus{suffix}.midx", "\n".join(lines))
        print("Midx Saved")

        print("Saving google")
        Visus.SaveTextDocument(f"{self.cache_dir}/google.midx",
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
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def debug_matches_graph(self):
        func_name = "debug_matches_graph"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        out = np.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        for bGoodMatches in [False, True]:
            for camera1 in self.cameras:
                local_cameras = camera1.getAllLocalCameras()
                for camera2 in local_cameras:
                    edge = camera1.getEdge(camera2)
                    if camera1.id < camera2.id and bGoodMatches == edge.isGood():
                        p0 = self.get_image_center(camera1)
                        p1 = self.get_image_center(camera2)
                        color = [211, 211, 211, 255]
                        if edge.isGood():
                            color = [0, 0, 0, 255]
                        cv2.line(out, p0, p1, color, 1)
                        num_matches = edge.matches.size()
                        if num_matches > 0:
                            cx = int(0.5 * (p0[0] + p1[0]))
                            cy = int(0.5 * (p0[1] + p1[1]))
                            cv2.putText(out, str(num_matches), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color)

        for camera in self.cameras:
            center = self.get_image_center(camera)
            cv2.putText(out, str(camera.id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                        [0, 0, 0, 255])

        img_utils.SaveImage(f"{self.cache_dir}/~matches.png", out)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def get_image_center(self, image):
        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        p = image.quad.centroid()
        return int((p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))

    def debug_solution(self):
        func_name = "debug_solution"
        start_time = None
        if self.verbose:
            start_time = time.time()

        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        out = np.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        for camera in self.cameras:
            color = (255 * camera.color.getRed(), 255 * camera.color.getGreen(), 255 * camera.color.getBlue(), 255)
            points = np.array([self.quad_to_screen(it, box) for it in camera.quad.points], dtype=np.int32)
            cv2.polylines(out, [points], True, color, 3)
            org = self.quad_to_screen(camera.quad.points[0], box)
            cv2.putText(out, str(camera.id), org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color)

        img_utils.SaveImage(f"{self.cache_dir}/~solution.png", out)

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def quad_to_screen(self, p, box):
        func_name2 = "quad_to_screen"
        start_time2 = None
        if self.verbose:
            start_time2 = time.time()

        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        ret_val = int(0 + (p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))

        if self.verbose:
            stop_time2 = time.time()
            if func_name2 not in self.execution_times:
                self.execution_times[func_name2] = []
            self.execution_times[func_name2].append(stop_time2 - start_time2)

        return ret_val

    def extract_key_points(self, image, camera):
        func_name = "extract_key_points"
        start_time = None
        if self.verbose:
            start_time = time.time()

        # Create idx and extract key points
        key_point_path = f"{self.cache_dir}/key_points/{camera.id}"
        idx_path = f"{self.cache_dir}/{camera.idx_filename}"

        if self.initial_multi_image:
            multi_image = self.initial_multi_image
            self.initial_multi_image = None
        else:
            multi_image = self.generate_multi_image(image)
        interleaved_image = self.interleave_channels(multi_image)

        data = Visus.LoadDataset(idx_path)
        #data.write(interleaved_image)

        # write zipped full
        data.compressDataset(["lz4"], Visus.Array.fromNumPy(interleaved_image, TargetDim=2, bShareMem=True))

        # if we're using micasense imagery, select the band specified by the user for extraction
        if self.multi_band:
            print(f"Using band index {self.extraction_band} for extraction")
            energy = interleaved_image[:, :, self.extraction_band]
        else:
            energy = cv2.cvtColor(interleaved_image, cv2.COLOR_RGB2GRAY)

        energy = cv2.resize(energy, self.energy_size)
        (key_points, descriptors) = self.extractor.doExtract(energy)

        if not self.vs:
            self.vs = self.width / float(energy.shape[1])

        if key_points:
            camera.keypoints.reserve(len(key_points))
            for p in key_points:
                kp = Visus.KeyPoint(self.vs * p.pt[0], self.vs * p.pt[1], p.size, p.angle, p.response, p.octave, p.class_id)
                camera.keypoints.push_back(kp)
            camera.descriptors = Visus.Array.fromNumPy(descriptors, TargetDim=2)
            super().saveKeyPoints(camera, key_point_path)

        print(f"Done converting and extracting {camera.filenames[0]}")

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

    def find_matches(self, camera1, camera2):
        func_name = "find_matches"
        start_time = None
        if self.verbose:
            start_time = time.time()

        if camera1.keypoints.empty() or camera2.keypoints.empty():
            camera2.getEdge(camera1).setMatches([], "No key points")
            return 0

        # We have already set matches for these two cameras
        if camera2.getEdge(camera1).isGood():
            return 0

        matches, H21, err = FindMatches(self.width, self.height,
                                        camera1.id, [(k.x, k.y) for k in camera1.keypoints],
                                        Visus.Array.toNumPy(camera1.descriptors),
                                        camera2.id, [(k.x, k.y) for k in camera2.keypoints],
                                        Visus.Array.toNumPy(camera2.descriptors),
                                        self.max_reprojection_error * self.width, self.ratio_check)

        if err:
            camera2.getEdge(camera1).setMatches([], err)
            return 0

        matches = [Visus.Match(match.queryIdx, match.trainIdx, match.imgIdx, match.distance) for match in matches]
        camera2.getEdge(camera1).setMatches(matches, str(len(matches)))

        if self.verbose:
            stop_time = time.time()
            if func_name not in self.execution_times:
                self.execution_times[func_name] = []
            self.execution_times[func_name].append(stop_time - start_time)

        return len(matches)

    def remove_bad_cameras(self):
        self.removeOutlierMatches(self.max_reprojection_error * self.width)
        self.removeDisconnectedCameras()
        self.removeCamerasWithTooMuchSkew()

    # Save all execution time data as a csv
    def write_times_to_csv(self, filename="times.csv"):
        if not self.execution_times.keys():
            return

        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "a") as file:
            file.write(f"function_name,total ({len(self.provider.images)} img),avg,min,max,\n")
            for key in self.execution_times.keys():
                times = self.execution_times[key]
                file.write(f"{key},")
                total = np.sum(times)
                file.write(f"{total},{total / len(times)},{np.min(times)},{np.max(times)},")
                for t in times:
                    file.write(f"{t},")
                file.write("\n")
            file.close()
