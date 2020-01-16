import skimage
import numpy
import skgeom
import shapely.geometry
import networkx
import sympy


def _find_contours(mask):
	contours = skimage.measure.find_contours(mask, 0.5)

	for pts in contours:
		if len(pts) < 3:
			continue

		if tuple(pts[0]) == tuple(pts[-1]):
			pts = pts[:-1]

		yield numpy.flip(pts, axis=-1)


def _point_to_tuple(p):
	return float(p.x()), float(p.y())


def _traceback(source, backward, node):
	yield node

	while node != source:
		node = backward[node]
		yield node


def _farthest(G, source):
	farthest = (0, None)

	len_up_to = dict(((source, 0),))
	backward = dict()
	for u, v, t in networkx.dfs_labeled_edges(G, source=source):
		if t == 'forward':
			u_v_len = numpy.linalg.norm(numpy.array(u) - numpy.array(v))
			path_len = len_up_to[u] + u_v_len
			len_up_to[v] = path_len

			backward[v] = u

			if path_len > farthest[0]:
				farthest = (path_len, v)

	_, node = farthest
	return list(reversed(list(_traceback(source, backward, node))))


def _leaf_nodes(G):
	degrees = numpy.array(list(G.degree(G.nodes)))
	return degrees[degrees[:, 1] == 1][:, 0]


def _longest_path(G):
	u = _farthest(G, _leaf_nodes(G)[0])[-1]
	return _farthest(G, u)


def _clip_path(origin, radius, path):
	path = numpy.array(path)
	prev_pt = origin
	while len(path) > 0:
		next_pt = path[0]
		d = numpy.linalg.norm(next_pt - origin)
		if d < radius:
			path = path[1:]
		else:
			intersections = sympy.intersection(
				sympy.Segment(prev_pt, next_pt),
				sympy.Circle(origin, radius))
			pt = intersections[0].evalf()
			path[0, :] = numpy.array([pt.x, pt.y], dtype=path.dtype)
			break

	return path


def _clip_path_2(path, radius):
	path = _clip_path(path[0], radius, path[1:])

	path = list(reversed(
		_clip_path(path[-1], radius, list(reversed(path[:-1])))))

	return path


class Polyline:
	def __init__(self, coords, width):
		self._coords = numpy.array(coords)
		self._width = width

	@property
	def coords(self):
		return self._coords

	def oriented(self, v):
	    u = self._coords[-1] - self._coords[0]
	    if numpy.dot(u, numpy.array(v)) < 0:
	        return Polyline(list(reversed(self._coords)), self._width)
	    else:
	        return self


def _find_polyline(contour):
	polygon = skgeom.Polygon(contour)

	if polygon.orientation() != skgeom.Sign.POSITIVE:
		return None

	skeleton = skgeom.skeleton.create_interior_straight_skeleton(polygon)

	G = networkx.Graph()
	G.add_nodes_from([_point_to_tuple(v.point) for v in skeleton.vertices])

	if len(G) < 2:
		return None

	G.add_edges_from([(
		_point_to_tuple(h.vertex.point),
		_point_to_tuple(h.opposite.vertex.point))
		for h in skeleton.halfedges if h.is_bisector])

	path = numpy.array(_longest_path(G))

	line_width = max(v.time for v in skeleton.vertices)

	return Polyline(_clip_path_2(path, line_width), line_width)


def from_mask(mask, orientation=None, k=1):
	polylines = []
	for contour in _find_contours(numpy.array(mask)):
		if k is not None:
			contour = numpy.array(
				shapely.geometry.LineString(contour).simplify(k).coords)
		polyline = _find_polyline(contour)
		if polyline:
			if orientation:
				polyline = polyline.oriented(orientation)
			polylines.append(polyline)
	return polylines


def from_palette_image(pixels, keys, orientations=None):
	pixels = numpy.array(pixels)

	results = dict()
	for key in keys:
		v = None
		if orientations:
			v = orientations.get(key, None)
		polylines = from_mask(pixels == key, v)
		results[key] = [p.coords for p in polylines]

	return results
