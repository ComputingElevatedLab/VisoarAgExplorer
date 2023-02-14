import logging
import math
import os.path
import random
import re
import time

import cv2
import numpy as np
from rtree import index as rtindex

import OpenVisus as Visus
import micasense.capture
import micasense.dls
import micasense.image
import micasense.panel
import slampy.image_provider as img_utils
from slampy.extract_keypoints import ExtractKeyPoints
from slampy.find_matches import FindMatches
from slampy.gps_utils import GPSUtils
from slampy.image_provider_generic import ImageProviderGeneric
from slampy.image_provider_lumenera import ImageProviderLumenera
from slampy.image_provider_micasense import ImageProviderRedEdge
from slampy.image_provider_sequoia import ImageProviderSequoia
from slampy.metadata_reader import MetadataReader


class Slam2DIncremental(Visus.Slam):
    def __init__(
        self,
        alt_threshold,
        colors,
        directory,
        output_dir,
        provider_type,
        extractor,
        extraction_band,
        resize_scale,
    ):
        logging.info("Initializing Slam2DIncremental object")
        super(Slam2DIncremental, self).__init__()
        self.depth = None
        self.width = 0
        self.height = 0
        self.dtype = Visus.DType()
        self.calibration = Visus.Calibration()
        self.image_dir = directory
        self.output_dir = output_dir
        self.thermal_dir = os.path.join(self.output_dir, "thermal")
        os.makedirs(self.thermal_dir, exist_ok=True)
        self.debug_mode = False
        self.energy_size = None
        self.physic_box = None
        self.enable_svg = colors
        self.enable_color_matching = False
        self.do_bundle_adjustment = True

        self.execution_times = {}
        self.extraction_band = extraction_band if extraction_band >= 1 else None
        self.world_centers = {}
        self.initial_multi_image = None
        self.initial_interleaved_image = None
        self.initial_generate_time = None
        self.lat0 = None
        self.lon0 = None
        self.alt0 = None
        self.plane = None
        self.vs = None
        self.distance_threshold = np.inf
        self.previous_yaw = 0
        self.reader = MetadataReader()
        self.idx_centers = rtindex.Index()
        self.idx_boxes = rtindex.Index()
        self.coordinates = {}
        self.align_pbox = None
        self.align_lbox = None
        self.translation_offset = None
        self.scale_offset = [1, 1]
        self.align_quad = None
        self.logic_box_string = ""
        self.cached_physic_box = None
        self.physic_box_string = ""
        self.physic_box_values = []
        self.resize_scale = resize_scale
        self.scale_values = None
        self.translate_values = None

        # Initialize the image provider
        self.alt_threshold = alt_threshold
        self.multi_band = False
        self.band_range = 0
        self.panels_found = False
        self.initialized = False
        self.provider_type = provider_type
        if self.provider_type == "lumenera":
            logging.info("Initializing Lumenera provider")
            self.provider = ImageProviderLumenera()
        elif self.provider_type == "rededge" or self.provider_type == "micasense":
            logging.info("Initializing Micasense provider")
            self.multi_band = True
            self.band_range = 6
            self.provider = ImageProviderRedEdge()
        elif self.provider_type == "altum-pt":
            logging.info("Initializing Altum-PT provider")
            self.multi_band = True
            self.band_range = 7
            self.provider = ImageProviderRedEdge()
        elif self.provider_type == "sequoia":
            logging.info("Initializing Sequoia provider")
            self.provider = ImageProviderSequoia()
        else:
            logging.info("Initializing generic provider")
            self.provider = ImageProviderGeneric()
        self.provider.skip_value = 1
        self.provider.telemetry = None
        self.provider.plane = None
        self.provider.image_dir = self.image_dir
        self.provider.cache_dir = self.output_dir
        self.provider.extractor_method = extractor

        self.min_key_points = 1000
        self.max_key_points = 6000
        self.anms = 2000
        self.max_reprojection_error = 0.01
        self.ratio_check = 0.8
        self.calibration.bFixed = False
        self.ba_tolerance = 0.005
        self.extractor = ExtractKeyPoints(
            self.min_key_points, self.max_key_points, self.anms, extractor, timing=True
        )

    def add_image(self, image_path):
        self.provider.addImage([image_path])
        image = self.provider.images[-1]
        self.load_image_metadata(image)

        if len(self.provider.images) == 1:
            self.initialize_slam(image)

        image.alt -= self.plane
        self.adjust_image_yaw(image)

        return image

    def add_multi_band_image(self, image_path):
        band = int(image_path[-5])
        image_bands = []
        for i in range(1, self.band_range + 1):
            if band == i:
                image_bands.append(image_path)
            else:
                ith_band_path = image_path[:-5] + str(i) + image_path[-4:]
                image_bands.append(ith_band_path)

        if not self.panels_found:
            panel_start = time.time()
            self.provider.addImage(image_bands)
            self.load_metadata(self.provider.images[-1])
            panel = micasense.panel.Panel(micasense.image.Image(image_bands[0]))
            if panel.panel_detected():
                panel_time = time.time() - panel_start
                return None, panel_time

            if len(self.provider.images) == 1:
                self.provider.images.pop(0)
                panel_time = time.time() - panel_start
                return None, panel_time

            self.provider.findPanels()
            self.panels_found = True
            panel_time = time.time() - panel_start
        else:
            self.provider.addImage(image_bands)
            panel_time = None

        if not self.provider.images:
            return None, panel_time

        image = self.provider.images[-1]
        self.load_image_metadata(image)

        if image.alt < self.alt_threshold:
            logging.info(
                f"Dropping image: altitude of {image.alt} is below the threshold {self.alt_threshold}"
            )
            self.provider.images.pop()
            return None, panel_time

        if not self.initialized:
            self.initialize_slam(image)
            self.initialized = True

        image.alt -= self.plane

        return image, panel_time

    def load_image_metadata(self, image):
        self.load_metadata(image)
        load_gps_from_metadata(image)
        self.load_yaw_from_metadata(image)

    def load_metadata(self, image):
        filename = image.filenames[0]
        image.metadata = self.reader.readMetadata(filename)

    def load_yaw_from_metadata(self, image):
        logging.info(f"Loading yaw from metadata")

        yaw = img_utils.FindMetadata(
            image.metadata, ["Yaw", "GimbalYaw", "GimbalYawDegree", "yaw", "yaw(gps)"]
        )

        if yaw in image.metadata:
            image.yaw = float(image.metadata[yaw])
            logging.info(f"yaw = {image.yaw}")
        else:
            image.yaw = self.previous_yaw
            logging.info(
                f"Did not find a yaw value, using the previous yaw={self.previous_yaw}"
            )

        # Convert the yaw to radians if necessary
        if not self.multi_band:
            if yaw and "radians" not in yaw.lower():
                image.yaw = np.radians(image.yaw)

    def interpolate_gps(self):
        n = len(self.provider.images)
        for i, image in enumerate(self.provider.images):
            if not wrong_gps(image):
                continue

            a = i - 1
            while a >= 0 and wrong_gps(self.provider.images[a]):
                a -= 1

            b = i + 1
            while b < n and wrong_gps(self.provider.images[b]):
                b += 1

            if 0 <= a < b < n:
                alpha = (i - a) / float(b - a)
                image.lat = (1 - alpha) * self.provider.images[
                    a
                ].lat + alpha * self.provider.images[b].lat
                image.lon = (1 - alpha) * self.provider.images[
                    a
                ].lon + alpha * self.provider.images[b].lon
                image.alt = (1 - alpha) * self.provider.images[
                    a
                ].alt + alpha * self.provider.images[b].alt
                logging.info(
                    f"Interpolating GPS: lat = {image.lat}, lon = {image.lon}, alt = {image.alt}"
                )
            else:
                raise Exception(
                    f"Error: could not interpolate GPS for {image.filenames[0]}"
                )

    def initialize_slam(self, image):
        if not image:
            return

        self.lat0 = image.lat
        self.lon0 = image.lon
        self.alt0 = image.alt
        self.plane = self.guess_plane(image)

        self.provider.loadSensorCfg()

        generate_start = time.time()
        self.initial_multi_image = self.generate_multi_image(image)
        self.initial_interleaved_image = interleave_channels(self.initial_multi_image)
        self.initial_generate_time = time.time() - generate_start

        image_as_array = Visus.Array.fromNumPy(
            self.initial_interleaved_image, TargetDim=2
        )

        self.width = image_as_array.getWidth()
        self.height = image_as_array.getHeight()
        self.depth = image_as_array.getDepth()
        self.dtype = image_as_array.dtype

        # Used to resize all incoming images
        energy_width = max(int(self.width * self.resize_scale), 256)
        energy_height = max(int(self.height * self.resize_scale), 256)
        self.energy_size = (energy_width, energy_height)

        # NOTE: from telemetry I'm just taking lat,lon,alt,yaw (not other stuff)
        if self.provider.telemetry:
            self.provider.loadTelemetry(self.provider.telemetry)
        if not self.provider.calibration:
            self.provider.calibration = self.provider.guessCalibration(
                self.initial_multi_image
            )

        # Needed so that RGB bands will be aligned later
        self.provider.findMultiAlignment(self.initial_multi_image)

        self.calibration = self.provider.calibration

    def generate_multi_image(self, image):
        return self.provider.generateMultiImage(image)

    def guess_plane(self, image):
        if self.multi_band:
            logging.info(f"Guessing plane from micasense panels")
            alt = img_utils.FindMetadata(image.metadata, ["GPSAltitude"])
            if alt:
                return np.min(
                    [float(panel.metadata[alt]) for panel in self.provider.panels]
                )

        absolute = img_utils.FindMetadata(
            image.metadata, ["AbsoluteAltitude", "GPSAltitude"]
        )
        relative = img_utils.FindMetadata(
            image.metadata, ["RelativeAltitude", "GPSAltitudeRef"]
        )

        if absolute and relative:
            return float(image.metadata[absolute]) - float(image.metadata[relative])

        return 0

    def adjust_image_yaw(self, image):
        if self.multi_band:
            image.yaw -= self.provider.yaw_offset / 2
        else:
            image.yaw += self.provider.yaw_offset
            offset = 2 * np.pi
            while image.yaw > np.pi:
                image.yaw -= offset
            while image.yaw < -np.pi:
                image.yaw += offset
        logging.info(
            f"{image.filenames[0]} - {image.yaw} radians {np.degrees(image.yaw)} degrees"
        )

    def add_camera(self, image):
        camera = Visus.Camera()
        camera.id = len(self.cameras)
        camera.color = Visus.Color.random()
        for filename in image.filenames:
            camera.filenames.append(filename)
        super().addCamera(camera)
        return self.cameras[-1]

    def create_idx(self, camera):
        camera.idx_filename = f"./idx/{camera.id:04d}.idx"
        field = Visus.Field("myfield", self.dtype)
        field.default_layout = "row_major"
        Visus.CreateIdx(
            url=os.path.join(self.output_dir, camera.idx_filename),
            dim=2,
            filename_template=f"./{camera.id:04d}.bin",
            blocksperfile=-1,
            fields=[field],
            dims=(self.width, self.height),
        )

    def set_initial_pose(self, image, camera):
        x, y = GPSUtils.gpsToLocalCartesian(image.lat, image.lon, self.lat0, self.lon0)
        self.world_centers[camera.id] = Visus.Point3d(x, y, image.alt)
        q = Visus.Quaternion(Visus.Point3d(0, 0, 1), -image.yaw) * Visus.Quaternion(
            Visus.Point3d(1, 0, 0), np.pi
        )
        camera.pose = Visus.Pose(q, self.world_centers[camera.id]).inverse()

    def refresh_quads(self):
        self.refreshQuads()

    def set_all_cameras(self, camera, n):
        center = self.world_centers[camera.id]
        camera.bFixed = False
        indices = list(set(self.idx_centers.nearest((center[0], center[1]), n)))

        for index in indices:
            other_camera = self.cameras[index]
            camera.addLocalCamera(other_camera)
            other_camera.addLocalCamera(camera)

    def set_local_cameras(self):
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
            logging.info(
                f"The following cameras are in the same position: {find_camera[center].id} {self.cameras[-1].id}"
            )
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
                for camera2 in adjacent[i + 1 :]:
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
            center = quad_to_screen(center, box)
            cell = np.array([quad_to_screen(it, box) for it in cell], dtype=np.int32)
            cv2.fillConvexPoly(
                out,
                cell,
                [
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ],
            )
            cv2.polylines(out, [cell], True, [0, 0, 0], 3)
            cv2.putText(
                out,
                str(camera2.id),
                (center[0], center[1]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.0,
                [0, 0, 0],
            )

    def get_distance_threshold(self, x, y):
        x_center = self.cameras[x].getWorldCenter()
        y_center = self.cameras[y].getWorldCenter()
        self.distance_threshold = self.cartesian_distance(x_center, y_center) * 1.5

    def insert_camera_center_into_spatial_index(self, camera):
        center = self.world_centers[camera.id]
        self.idx_centers.insert(camera.id, (center[0], center[1]))

    def insert_camera_box_into_spatial_index(self, camera):
        box = camera.quad.getBoundingBox()
        self.idx_boxes.insert(camera.id, [box.p1[0], box.p1[1], box.p2[0], box.p2[1]])

    def select_and_match_indices(self, index, method=0):
        if method == 0:
            # rtree select closest N
            # Total elapsed time: 634.3631505966187
            logging.info(f"Getting intersecting indices on index {index}")
            indices = self.get_nearest_n_indices(index, 50)
            logging.info(f"Number of images being bundle adjusted: {len(indices)}")
            self.find_matches_among_indices(indices)
        elif method == 1:
            # Match all to all but set only some as bFixed
            # Total elapsed time: 644.3396918773651 s
            camera_i = self.cameras[-1]
            for camera_j in self.cameras:
                if camera_i == camera_j:
                    continue
                if camera_i.getEdge(camera_j) is not None:
                    self.find_matches(camera_i, camera_j)
            logging.info(f"Getting intersecting indices on index {index}")
            self.get_nearest_n_indices(index, 50)
        elif method == 2:
            # Match all to all but set only some as bFixed
            # Total elapsed time: 720.8694500923157 s
            camera_i = self.cameras[-1]
            for camera_j in self.cameras:
                if camera_i == camera_j:
                    continue
                if camera_i.getEdge(camera_j) is not None:
                    self.find_matches(camera_i, camera_j)
            logging.info(f"Getting intersecting indices on index {index}")
            self.get_intersecting_indices(index)
        elif method == 3:
            # Always match all to all (full global badj)
            # Total elapsed time: 836.8993334770203 s
            camera_i = self.cameras[-1]
            for camera_j in self.cameras:
                if camera_i == camera_j:
                    continue
                if camera_i.getEdge(camera_j) is not None:
                    self.find_matches(camera_i, camera_j)
        elif method == 4:
            # rtree select intersecting
            # Total elapsed time: 2349.37468457222
            logging.info(f"Getting intersecting indices on index {index}")
            indices = self.get_intersecting_indices(index)
            logging.info(f"Number of images being bundle adjusted: {len(indices)}")
            self.find_matches_among_indices(indices)

    def get_intersecting_indices(self, at):
        camera = self.cameras[at]
        box = camera.quad.getBoundingBox()
        camera.bFixed = False
        indices = list(
            set(
                self.idx_boxes.intersection(
                    [box.p1[0], box.p1[1], box.p2[0], box.p2[1]]
                )
            )
        )

        for i, other_camera in enumerate(self.cameras):
            if i == camera.id:
                continue
            elif i in indices:
                other_camera.color = camera.color
                other_camera.bFixed = False
            else:
                other_camera.bFixed = True

        return indices

    def get_nearest_n_indices(self, at, n):
        camera = self.cameras[at]
        center = self.world_centers[camera.id]
        camera.bFixed = False
        indices = list(set(self.idx_centers.nearest((center[0], center[1]), n)))

        for i, other_camera in enumerate(self.cameras):
            if i == camera.id:
                continue
            elif i in indices:
                other_camera.color = camera.color
                other_camera.bFixed = False
            else:
                other_camera.bFixed = True

        return indices

    def find_matches_among_indices(self, indices):
        for i, v in enumerate(indices):
            camera_i = self.cameras[v]
            for k in range(i + 1, len(indices)):
                j = indices[k]
                camera_j = self.cameras[j]
                if camera_i.getEdge(camera_j) is not None:
                    self.find_matches(camera_i, camera_j)

    def find_nearest_n_matches(self, camera, n):
        center = self.world_centers[camera.id]
        nearby = list(set(self.idx_centers.nearest((center[0], center[1]), n)))

        for i, v in enumerate(nearby):
            if v == camera.id:
                continue
            other_camera = self.cameras[v]
            if camera.getEdge(other_camera) is not None:
                self.find_matches(camera, other_camera)

    def get_local_ba_indices(self, at, previous):
        if previous:
            self.get_distance_threshold(at, previous[0])
        else:
            self.get_distance_threshold(0, -1)

        indices = [at]
        camera = self.cameras[at]
        camera.bFixed = False
        for i, other_camera in enumerate(self.cameras):
            if i == at:
                continue

            distance = self.cartesian_distance(
                self.world_centers[camera.id], self.world_centers[other_camera.id]
            )
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

                distance = self.cartesian_distance(
                    self.world_centers[camera.id], self.world_centers[other_camera.id]
                )
                if distance > self.distance_threshold:
                    continue

                if Visus.Quad.intersection(camera.quad, other_camera.quad):
                    self.find_matches(camera, other_camera)
                    other_camera.bFixed = False

        return indices

    # Cartesian distance between two (x, y, z) coordinates
    @staticmethod
    def cartesian_distance(p1, p2):
        return (
            ((p2.x - p1.x) ** 2) + ((p2.y - p1.y) ** 2) + ((p2.z - p1.z) ** 2)
        ) ** 0.5

    # Cartesian distance between two (x, y, z) coordinates
    @staticmethod
    def cartesian_distance_from_array(p1, p2):
        return (((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2)) ** 0.5

    def bundle_adjust(self, algorithm=""):
        start_time = time.time()
        self.bundleAdjustment(self.ba_tolerance, algorithm)
        return time.time() - start_time

    def create_visus_midx(self, suffix=""):
        if self.translation_offset:
            logging.info(f"Aligning visus{suffix}.midx to Google Maps")
        else:
            logging.info(f"Creating visus{suffix}.midx")

        logic_box = self.getQuadsBox()

        # instead of working in range -180,+180 -90,+90 (worldwise ref frame) I normalize the range in [0,1]*[0,1]
        # physic box is in the range [0,1]*[0,1]
        # logic_box is in pixel coordinates
        # NOTE: I can override the physic box by command line
        physic_box = self.physic_box
        if physic_box is not None:
            logging.info("Using physic_box forced by the user", physic_box.toString())
        else:
            physic_box = Visus.BoxNd.invalid()
            for camera in self.cameras:
                quad = self.computeWorldQuad(camera)
                if self.translation_offset is not None:
                    quad = quad.translate(self.translation_offset)
                for point in quad.points:
                    lat, lon = GPSUtils.localCartesianToGps(
                        point.x, point.y, self.lat0, self.lon0
                    )
                    alpha, beta = GPSUtils.gpsToUnit(lat, lon)
                    physic_box.addPoint(Visus.PointNd(Visus.Point2d(alpha, beta)))
            self.cached_physic_box = physic_box

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
        self.logic_box_string = (
            f"{int(logic_box.p1[0])} {int(logic_box.p2[0])} "
            f"{int(logic_box.p1[1])} {int(logic_box.p2[1])}"
        )
        self.physic_box_string = f"{physic_box.p1[0]} {physic_box.p2[0]} {physic_box.p1[1]} {physic_box.p2[1]}"
        lines = [
            f'<dataset typename="IdxMultipleDataset" '
            f'logic_box="{self.logic_box_string}" physic_box="{self.physic_box_string}">',
            '\t<slam width="%s" height="%s" dtype="%s" calibration="%s %s %s"/>'
            % (
                Visus.cstring(self.width),
                Visus.cstring(self.height),
                self.dtype.toString(),
                Visus.cstring(self.calibration.f),
                Visus.cstring(self.calibration.cx),
                Visus.cstring(self.calibration.cy),
            ),
        ]

        # If we're using a micasense camera, create a field for each band
        if self.multi_band:
            lines.append("\t<field name='rgb'>")
            lines.append(
                "\t\t<code>output=ArrayUtils.interleave(ArrayUtils.split(voronoi())[0:3])</code>"
            )
            lines.append("\t</field>")

            for i in range(self.band_range):
                lines.append(f'\t<field name="band{i}">')
                lines.append(
                    f"\t\t<code>output=ArrayUtils.split(voronoi())[{i}]</code>"
                )
                lines.append(f"\t</field>")
        else:
            lines.append("\t<field name='voronoi'><code>")
            lines.append("\t\toutput=voronoi()</code>")
            lines.append("\t</field>")

        # how to go from logic_box (i.e. pixel) -> physic box ([0,1]*[0,1])
        self.translate_values = [physic_box.p1[0], physic_box.p1[1]]
        self.scale_values = [
            physic_box.size()[0] / logic_box.size()[0],
            physic_box.size()[1] / logic_box.size()[1],
        ]
        lines.append(
            '\t<translate x="%s" y="%s">'
            % (Visus.cstring10(physic_box.p1[0]), Visus.cstring10(physic_box.p1[1]))
        )
        lines.append(
            '\t\t<scale x="%s" y="%s">'
            % (
                Visus.cstring10(self.scale_values[0] * self.scale_offset[0]),
                Visus.cstring10(self.scale_values[1] * self.scale_offset[1]),
            )
        )
        lines.append(
            '\t\t\t<translate x="%s" y="%s">'
            % (Visus.cstring10(-logic_box.p1[0]), Visus.cstring10(-logic_box.p1[1]))
        )

        if self.enable_svg:
            W = int(1048)
            H = int(W * (logic_box.size()[1] / float(logic_box.size()[0])))

            lines.append(
                '\t\t\t\t<svg width="%s" height="%s" viewBox="%s %s %s %s">'
                % (
                    Visus.cstring(W),
                    Visus.cstring(H),
                    Visus.cstring(int(logic_box.p1[0])),
                    Visus.cstring(int(logic_box.p1[1])),
                    Visus.cstring(int(logic_box.p2[0])),
                    Visus.cstring(int(logic_box.p2[1])),
                )
            )

            lines.append(
                '\t\t\t\t\t<g stroke="#000000" stroke-width="1" fill="#ffff00" fill-opacity="0.3">'
            )
            for i, camera in enumerate(self.cameras):
                lines.append(
                    '\t\t\t\t\t\t<poi point="%s,%s" />'
                    % (
                        Visus.cstring(logic_centers[i][0]),
                        Visus.cstring(logic_centers[i][1]),
                    )
                )
            lines.append("\t\t\t\t\t</g>")

            # lines.append("<g fill-opacity='0.0' stroke-opacity='0.5' stroke-width='2'>")
            # for camera in self.cameras:
            #     lines.append("\t<polygon points='%s' stroke='%s' />" % (
            #         camera.quad.toString(",", " "), camera.color.toString()[0:7]))
            # lines.append("</g>")

            lines.append("\t\t\t\t</svg>")

        for camera in self.cameras:
            p = camera.getWorldCenter()
            lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, self.lat0, self.lon0)
            lines.append(
                '\t\t\t\t<dataset url="%s" color="%s" quad="%s" filenames="%s" q="%s" t="%s" lat="%s" lon="%s" '
                'alt="%s" />'
                % (
                    f"./{camera.idx_filename[2:]}",
                    camera.color.toString(),
                    camera.quad.toString(),
                    ";".join(camera.filenames),
                    camera.pose.q.toString(),
                    camera.pose.t.toString(),
                    Visus.cstring10(lat),
                    Visus.cstring10(lon),
                    Visus.cstring10(p.z),
                )
            )

        lines.append("\t\t\t</translate>")
        lines.append("\t\t</scale>")
        lines.append("\t</translate>")
        lines.append("\t<idxfile>")
        lines.append('\t\t<version value="6" />')
        lines.append('\t\t<bitmask value="V00101010101010101010101" />')
        lines.append(
            f'\t\t<box value="{int(logic_box.p1[0])} {int(logic_box.p2[0])} '
            f'{int(logic_box.p1[1])} {int(logic_box.p2[1])}" />'
        )
        lines.append('\t\t<bitsperblock value="16" />')
        lines.append('\t\t<blocksperfile value="21" />')
        lines.append('\t\t<block_interleaving value="0" />')
        lines.append(f'\t\t<filename_template value="./visus{suffix}/%04x.bin" />')
        lines.append('\t\t<missing_blocks value="False" />')
        lines.append('\t\t<time_template value="" />')
        lines.append(
            '\t\t<logic_to_physic value="1.14802e-09 0 0.167416 0 1.15144e-09 0.64541 0 0 1" />'
        )
        for i in range(self.band_range):
            lines.append(
                f'\t\t<field name="band{i}" description="" index="{i}" default_compression="" default_layout="" '
                f'default_value="0" filter="" dtype="float32" />'
            )
        lines.append('\t\t<timestep when="0" />')
        lines.append("\t</idxfile>")
        lines.append("</dataset>")

        Visus.SaveTextDocument(
            f"{self.output_dir}/visus{suffix}.midx", "\n".join(lines)
        )
        if self.translation_offset:
            logging.info(f"Aligned visus{suffix}.midx to Google Maps")
        else:
            logging.info(f"Created visus{suffix}.midx")

    def create_single_image_midx(self):
        logging.info(f"Creating visus_initial.midx")

        camera = self.cameras[0]

        logic_box = camera.quad.getBoundingBox()

        # instead of working in range -180,+180 -90,+90 (worldwise ref frame) I normalize the range in [0,1]*[0,1]
        # physic box is in the range [0,1]*[0,1]
        # logic_box is in pixel coordinates
        # NOTE: I can override the physic box by command line
        physic_box = self.physic_box
        if physic_box is not None:
            logging.info("Using physic_box forced by the user", physic_box.toString())
        else:
            physic_box = Visus.BoxNd.invalid()
            quad = self.computeWorldQuad(camera)
            for point in quad.points:
                lat, lon = GPSUtils.localCartesianToGps(
                    point.x, point.y, self.lat0, self.lon0
                )
                alpha, beta = GPSUtils.gpsToUnit(lat, lon)
                physic_box.addPoint(Visus.PointNd(Visus.Point2d(alpha, beta)))

        logic_centers = []
        p = camera.getWorldCenter()
        lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, self.lat0, self.lon0)
        alpha, beta = GPSUtils.gpsToUnit(lat, lon)
        alpha = (alpha - physic_box.p1[0]) / float(physic_box.size()[0])
        beta = (beta - physic_box.p1[1]) / float(physic_box.size()[1])
        logic_x = logic_box.p1[0] + alpha * logic_box.size()[0]
        logic_y = logic_box.p1[1] + beta * logic_box.size()[1]
        logic_centers.append((logic_x, logic_y))

        # This is the midx, dump some information about the slam
        lines = [
            "<dataset typename='IdxMultipleDataset' logic_box='%s %s %s %s' physic_box='%s %s %s %s'>"
            % (
                Visus.cstring(int(logic_box.p1[0])),
                Visus.cstring(int(logic_box.p2[0])),
                Visus.cstring(int(logic_box.p1[1])),
                Visus.cstring(int(logic_box.p2[1])),
                Visus.cstring10(physic_box.p1[0]),
                Visus.cstring10(physic_box.p2[0]),
                Visus.cstring10(physic_box.p1[1]),
                Visus.cstring10(physic_box.p2[1]),
            ),
            "",
            "<slam width='%s' height='%s' dtype='%s' calibration='%s %s %s' />"
            % (
                Visus.cstring(self.width),
                Visus.cstring(self.height),
                self.dtype.toString(),
                Visus.cstring(self.calibration.f),
                Visus.cstring(self.calibration.cx),
                Visus.cstring(self.calibration.cy),
            ),
            "",
            "<field name='voronoi'><code>output=voronoi()</code></field>",
            "",
            "<translate x='%s' y='%s'>"
            % (Visus.cstring10(physic_box.p1[0]), Visus.cstring10(physic_box.p1[1])),
            "<scale     x='%s' y='%s'>"
            % (
                Visus.cstring10((physic_box.size()[0]) / logic_box.size()[0]),
                Visus.cstring10((physic_box.size()[1]) / logic_box.size()[1]),
            ),
            "<translate x='%s' y='%s'>"
            % (Visus.cstring10(-logic_box.p1[0]), Visus.cstring10(-logic_box.p1[1])),
            "",
        ]

        # how to go from logic_box (i.e. pixel) -> physic box ([0,1]*[0,1])

        if self.enable_svg:
            W = int(1048)
            H = int(W * (logic_box.size()[1] / float(logic_box.size()[0])))

            lines.append(
                "<svg width='%s' height='%s' viewBox='%s %s %s %s' >"
                % (
                    Visus.cstring(W),
                    Visus.cstring(H),
                    Visus.cstring(int(logic_box.p1[0])),
                    Visus.cstring(int(logic_box.p1[1])),
                    Visus.cstring(int(logic_box.p2[0])),
                    Visus.cstring(int(logic_box.p2[1])),
                )
            )

            lines.append(
                "<g stroke='#000000' stroke-width='1' fill='#ffff00' fill-opacity='0.3'>"
            )
            lines.append(
                "\t<poi point='%s,%s' />"
                % (
                    Visus.cstring(logic_centers[0][0]),
                    Visus.cstring(logic_centers[0][1]),
                )
            )
            lines.append("</g>")

            lines.append("<g fill-opacity='0.0' stroke-opacity='0.5' stroke-width='2'>")
            lines.append(
                "\t<polygon points='%s' stroke='%s' />"
                % (camera.quad.toString(",", " "), camera.color.toString()[0:7])
            )
            lines.append("</g>")

            lines.append("</svg>")
            lines.append("")

        p = camera.getWorldCenter()
        lat, lon = GPSUtils.localCartesianToGps(p.x, p.y, self.lat0, self.lon0)
        lines.append(
            "<dataset url='%s' color='%s' quad='%s' filenames='%s' q='%s' t='%s' lat='%s' lon='%s' alt='%s' />"
            % (
                camera.idx_filename,
                camera.color.toString(),
                camera.quad.toString(),
                ";".join(camera.filenames),
                camera.pose.q.toString(),
                camera.pose.t.toString(),
                Visus.cstring10(lat),
                Visus.cstring10(lon),
                Visus.cstring10(p.z),
            )
        )

        lines.append("")
        lines.append("</translate>")
        lines.append("</scale>")
        lines.append("</translate>")
        lines.append("")
        lines.append("</dataset>")

        Visus.SaveTextDocument(
            f"{self.output_dir}/visus_initial.midx", "\n".join(lines)
        )
        logging.info(f"Created visus_initial.midx")

    def create_google_no_dataset_midx(self):
        logging.info("Creating google-no-dataset.midx")

        with open(f"{self.output_dir}/google-no-dataset.midx", "w+") as outfile:
            with open("templates/google-no-dataset.midx") as infile:
                outfile.write(infile.read())

        logging.info("Created google-no-dataset.midx")

    def create_google(self, suffix=""):
        if self.multi_band:
            self.create_google_xml(suffix)
        else:
            self.create_google_midx(suffix)

    def create_google_midx(self, suffix=""):
        logging.info("Creating google.midx")
        lines = [
            "<dataset name='slam' typename='IdxMultipleDataset'>",
            "<field name='voronoi'><code>output=voronoi()</code></field>",
            "\t<dataset typename='GoogleMapsDataset' tiles='https://mt1.google.com/vt/lyrs=s' physic_box='0.0 1.0 "
            "0.0 1.0' />",
            f"\t<dataset name='visus'   url='./visus{suffix}.midx' />",
            "</dataset>",
        ]
        Visus.SaveTextDocument(f"{self.output_dir}/google.midx", "\n".join(lines))
        logging.info("Created google.midx")

    def create_google_xml(self, suffix=""):
        logging.info("Creating google-multiband.xml")

        with open(f"{self.output_dir}/google-multiband.xml", "w+") as xml:
            with open("templates/google-xml-header-1.txt") as header:
                xml.write(header.read())
                xml.write(
                    f'\t\t\t<GLOrthoCamera pos="0.000000 0.000000 0.000000" center="0.000000 0.000000 -1.000000" '
                    f'vup="0.000000 1.000000 0.000000" rotation="0.000000" ortho_params="'
                    f"{self.translate_values[0] - 0.000005} {self.translate_values[0] + 0.000014} "
                    f'{self.translate_values[1] - 0.000005} {self.translate_values[1] + 0.000014} 1.000000 -1.000000" '
                    f'default_scale="1.300000" disable_rotation="False" max_zoom="0.000000" min_zoom="0.000000" '
                    f'default_smooth="500" />\n'
                )
                xml.write("\t\t</GLCameraNode>\n")
                xml.write("\t</AddNode>\n")
                xml.write('\t<AddNode parent="world">\n')

            path = os.path.abspath(self.output_dir)
            xml.write(
                f'\t\t<DatasetNode uuid="dataset" name="file://{path}/align.midx" visible="True" show_bounds="True">\n'
            )
            xml.write(
                f'\t\t\t<dataset url="file://{path}/align.midx" name="slam" typename="IdxMultipleDataset">\n'
            )

            with open("templates/google-xml-header-2.txt") as header:
                xml.write(header.read())

            xml.write(
                f'\t\t<DatasetNode uuid="dataset1" name="file://{path}/visus{suffix}.midx" visible="True" '
                f'show_bounds="True">\n'
            )

            with open(f"{path}/visus{suffix}.midx") as midx:
                midx.readline()
                xml.write(
                    f'\t\t\t<dataset url="file://{path}/visus{suffix}.midx" typename="IdxMultipleDataset" '
                    f'logic_box="{self.logic_box_string}" physic_box="{self.physic_box_string}">\n'
                )
                for line in midx:
                    xml.write(f"\t\t\t{line}")
                xml.write("\n")

            with open("templates/google-xml-footer-1.txt") as footer:
                xml.write(footer.read())

            xml.write(
                f'\t\t\t<node_bounds T="{self.scale_values[0]} 0 {self.translate_values[0]} 0 {self.scale_values[1]} '
                f'{self.translate_values[1]} 0 0 1" box="{self.logic_box_string}" />\n'
            )

            with open("templates/google-xml-footer-2.txt") as footer:
                xml.write(footer.read())

        logging.info("Created google-multiband.xml")

    def debug_matches_graph(self):
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
                            cv2.putText(
                                out,
                                str(num_matches),
                                (cx, cy),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1.0,
                                color,
                            )

        for camera in self.cameras:
            center = self.get_image_center(camera)
            cv2.putText(
                out,
                str(camera.id),
                (center[0], center[1]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.0,
                [0, 0, 0, 255],
            )

        img_utils.SaveImage(f"{self.output_dir}/~matches.png", out)

    def get_image_center(self, image):
        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        p = image.quad.centroid()
        return int((p[0] - box.p1[0]) * (W / w)), int(H - (p[1] - box.p1[1]) * (H / h))

    def debug_solution(self):
        box = self.getQuadsBox()
        w = float(box.size()[0])
        h = float(box.size()[1])
        W = int(4096)
        H = int(h * (W / w))
        out = np.zeros((H, W, 4), dtype="uint8")
        out.fill(255)

        for camera in self.cameras:
            color = (
                255 * camera.color.getRed(),
                255 * camera.color.getGreen(),
                255 * camera.color.getBlue(),
                255,
            )
            points = np.array(
                [quad_to_screen(it, box) for it in camera.quad.points],
                dtype=np.int32,
            )
            cv2.polylines(out, [points], True, color, 3)
            org = quad_to_screen(camera.quad.points[0], box)
            cv2.putText(
                out, str(camera.id), org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, color
            )

        img_utils.SaveImage(f"{self.output_dir}/~solution.png", out)

    def interleave_and_write_image(self, image, camera):
        if self.initial_multi_image:
            interleaved_image = self.initial_interleaved_image
            self.initial_multi_image = None
            self.initial_interleaved_image = None
            generate_time = self.initial_generate_time
        else:
            generate_start = time.time()
            multi_image = self.generate_multi_image(image)
            interleaved_image = interleave_channels(multi_image)
            generate_time = time.time() - generate_start
            logging.info(f"Interleaved image generated in: {generate_time} s")

        dataset_start = time.time()
        data = Visus.LoadDataset(f"{self.output_dir}/{camera.idx_filename}")
        data.write(interleaved_image)
        write_time = time.time() - dataset_start
        logging.info(f"Wrote interleaved image data in: {write_time} s")

        return interleaved_image, generate_time, write_time

    def extract_key_points(self, interleaved_image, camera):
        start_time = time.time()

        # if we're using micasense imagery, select the band specified by the user for extraction
        if self.multi_band:
            logging.info(f"Using band index {self.extraction_band} for extraction")
            energy = interleaved_image[:, :, self.extraction_band - 1]
        else:
            energy = cv2.cvtColor(interleaved_image, cv2.COLOR_RGB2GRAY)

        energy = cv2.resize(energy, self.energy_size)

        feature_start = time.time()
        key_points, descriptors, extract_time = self.extractor.doExtract(energy)
        feature_time = time.time() - feature_start
        logging.info(f"Extracted key points in: {feature_time} s")

        if not self.vs:
            self.vs = self.width / float(energy.shape[1])

        if key_points:
            camera.keypoints.reserve(len(key_points))
            for p in key_points:
                kp = Visus.KeyPoint(
                    self.vs * p.pt[0],
                    self.vs * p.pt[1],
                    p.size,
                    p.angle,
                    p.response,
                    p.octave,
                    p.class_id,
                )
                camera.keypoints.push_back(kp)
            camera.descriptors = Visus.Array.fromNumPy(descriptors, TargetDim=2)
            super().saveKeyPoints(camera, f"{self.output_dir}/key_points/{camera.id}")

        convert_time = time.time() - start_time
        logging.info(
            f"Done extracting and converting key points from {camera.filenames[0]} in {convert_time} s"
        )

        return convert_time, feature_time, extract_time, len(key_points)

    def find_matches(self, camera1, camera2):
        start_time = time.time()

        if camera1.keypoints.empty() or camera2.keypoints.empty():
            camera2.getEdge(camera1).setMatches([], "No key points")
            return 0

        # We have already set matches for these two cameras
        if camera2.getEdge(camera1).isGood():
            return 0

        matches, h, err = FindMatches(
            self.width,
            self.height,
            camera1.id,
            [(k.x, k.y) for k in camera1.keypoints],
            Visus.Array.toNumPy(camera1.descriptors),
            camera2.id,
            [(k.x, k.y) for k in camera2.keypoints],
            Visus.Array.toNumPy(camera2.descriptors),
            self.max_reprojection_error * self.width,
            self.ratio_check,
        )

        if err:
            camera2.getEdge(camera1).setMatches([], err)
            return 0

        matches = [
            Visus.Match(match.queryIdx, match.trainIdx, match.imgIdx, match.distance)
            for match in matches
        ]
        num_matches = len(matches)
        camera1.getEdge(camera2).setMatches(matches, str(num_matches))
        camera2.getEdge(camera1).setMatches(matches, str(num_matches))

        execution_time = time.time() - start_time
        logging.info(
            f"Found num_matches({num_matches}) matches in ms: {execution_time * 1000}"
        )

        return num_matches

    def find_all_matches(self):
        n = len(self.cameras)
        for i, camera_i in enumerate(self.cameras):
            for j in range(i + 1, n):
                camera_j = self.cameras[j]
                self.find_matches(camera_i, camera_j)

    def remove_bad_cameras(self):
        logging.info("Removing outlier matches")
        self.removeOutlierMatches(self.max_reprojection_error * self.width)
        logging.info("Removing disconnected cameras")
        self.removeDisconnectedCameras()
        logging.info("Removing cameras with too much skew")
        self.removeCamerasWithTooMuchSkew()

    def blend(self, count):
        f = open(f"{self.output_dir}/visus{count}.midx", "r")
        f = f.read()

        quad = r'<dataset\b[^<>]* quad="([^"]+)"'
        m = re.findall(quad, f)
        quad_points = []
        for i in m:
            float_list = list(map(float, i.split(" ")))
            pair_list = []
            for x in zip(float_list[::2], float_list[1::2]):
                pair_list.append(x)
            quad_points.append(list(pair_list))

        corners = []
        maxy = 0
        warped = []
        masks = []
        sizes = []
        for i, points in enumerate(quad_points):
            tlx = int(min(points[0][0], points[1][0], points[2][0], points[3][0]))
            tly = int(min(points[0][1], points[1][1], points[2][1], points[3][1]))
            blx = int(max(points[0][0], points[1][0], points[2][0], points[3][0]))
            bly = int(max(points[0][1], points[1][1], points[2][1], points[3][1]))

            if bly > maxy:
                maxy = bly

            img = cv2.imread(self.cameras[i].filenames[0])

            width, height = img.shape[1], img.shape[0]
            sizes.append((img.shape[1], img.shape[0]))

            newpoints = [[p[0] - tlx, p[1] - tly] for p in points]

            corners.append([tlx, tly])

            maskarray = 255 * np.ones(img.shape, dtype="uint8")

            pts1 = np.float32(
                [
                    list(newpoints[0]),
                    list(newpoints[1]),
                    list(newpoints[2]),
                    list(newpoints[3]),
                ]
            )
            pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warp = cv2.warpPerspective(img, matrix, (width, height))
            warped_mask = cv2.warpPerspective(maskarray, matrix, (width, height))
            mask = cv2.cvtColor(warped_mask, cv2.COLOR_RGB2GRAY)
            warped.append(warp)
            masks.append(mask)

        for c in corners:
            c[1] = maxy - c[1]

        seam_finder = cv2.detail_DpSeamFinder("COLOR")
        masks_seams = seam_finder.find(warped, corners, masks)

        masks_seams = list(masks_seams)

        blend_strength = 5
        blend_type = "feather"

        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
        if blend_width < 1:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
        elif blend_type == "multiband":
            blender = cv2.detail_MultiBandBlender()
            blender.setNumBands(
                (np.log(blend_width) / np.log(2.0) - 1.0).astype(np.int32)
            )
        elif blend_type == "feather":
            blender = cv2.detail_FeatherBlender()
            blender.setSharpness(1.0 / blend_width)
        blender.prepare(dst_sz)

        for i in range(len(masks)):
            image_warped_s = warped[i].astype(np.int16)
            masks_seams[i] = cv2.convertScaleAbs(masks_seams[i])
            blender.feed(cv2.UMat(image_warped_s), masks_seams[i], corners[i])

        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        dst = cv2.normalize(
            src=result,
            dst=None,
            alpha=255.0,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        cv2.imwrite(f"{self.output_dir}/result.jpg", dst)


def load_gps_from_metadata(image):
    logging.info("Loading GPS latitude, longitude, and altitude from metadata")

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

    logging.info(f"lat = {image.lat}, lon = {image.lon}, alt = {image.alt}")


def wrong_gps(image):
    return image.lat == 0.0 or image.lon == 0.0 or image.alt == 0.0


def interleave_channels(multi_image):
    if len(multi_image) == 1:
        return multi_image[0]
    return np.dstack(multi_image)


def quad_to_screen(p, box):
    w0 = float(box.size()[0])
    h0 = float(box.size()[1])
    w1 = int(4096)
    h1 = int(h0 * (w1 / w0))
    return int(0 + (p[0] - box.p1[0]) * (w1 / w0)), int(
        h1 - (p[1] - box.p1[1]) * (h1 / h0)
    )
