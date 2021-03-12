#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb-master/stb_image_write.h>
#include <Eigen/Dense>

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

#include "omp.h"

using vec3f = Eigen::Matrix<float, 3, 1>;
using point = Eigen::Matrix<float, 3, 1>;
using color = Eigen::Matrix<float, 3, 1>;

std::default_random_engine e;
std::uniform_real_distribution<float> u(0.0f, 1.0f);

const float MAX = 1000.0f;
const float MIN = 0.001f;

struct pixel {
	unsigned char r, g, b;
};

pixel* data = nullptr;

struct ray {
	ray() = default;
	ray(point _origin, vec3f _dir) : origin(_origin), dir(_dir){}

	point origin;
	vec3f dir;
};


struct hit_record {
	float t;
	vec3f n;
};

class material {
public:
	material() = default;
	material(color _albedo) : albedo(_albedo) {}
	virtual void scattar(const ray& r, const hit_record& rec, color* attenuation, ray* scattered) = 0;

	color albedo;
};

class lambertian : public material {
public:
	lambertian() = default;
	lambertian(color _albedo) : material(_albedo) {}

	virtual void scattar(const ray& r, const hit_record& rec, color* attenuation, ray* scattered) {
		point surface_point = r.origin + r.dir * rec.t;
		point tf(u(e) - .5, u(e) - .5, u(e) - .5);
		tf.normalize();
		point target = surface_point + rec.n + tf;
		vec3f dir = target - surface_point;
		
		scattered->origin = surface_point;
		scattered->dir = dir;

		*attenuation = albedo;
	}
};

class metal : public material {
public:
	metal() = default;
	metal(color _albedo) : material(_albedo) {}

	virtual void scattar(const ray& r, const hit_record& rec, color* attenuation, ray* scattered) {
		point surface_point = r.origin + r.dir * rec.t;
		vec3f reflect = r.dir - 2 * r.dir.dot(rec.n) * rec.n;
		scattered->origin = surface_point;
		scattered->dir = reflect;
		*attenuation = albedo;
	}
};

class luminous : public material {
public:
	luminous() = default;
	luminous(color _albedo) : material(_albedo) {}

	virtual void scattar(const ray& r, const hit_record& rec, color* attenuation, ray* scattered) {
		point surface_point = r.origin + r.dir * rec.t;
		vec3f reflect = vec3f(0, 0, 0);
		scattered->origin = surface_point;
		scattered->dir = reflect;
		*attenuation = albedo;
	}
};

class hittable {
public:
	hittable() = default;
	hittable(material* _m) : m(_m) {}
	virtual bool hit(ray& r, hit_record* rec) = 0;

	material* m;
};

class sphere : public hittable{
public:
	sphere() = default;
	sphere(point _center, float _radius, material* _m) : center(_center), radius(_radius), hittable(_m){}

	virtual bool hit(ray& r, hit_record* rec) {
		double a, b, c;
		double dsc, rt, sqd;
		vec3f x, p, nor;
		x = center - r.origin;
		p = r.dir;
		a = p.squaredNorm();
		b = -2 * (x.dot(p));
		c = x.squaredNorm() - radius * radius;
		dsc = b * b - 4 * a * c;
		sqd = std::sqrt(dsc);
		rt = (-b - sqd) / (2 * a);
		if (rt > MAX || rt < MIN) {
			rt = (-b + sqd) / (2 * a);
			if (rt > MAX || rt < MIN)
				return false;
		}
		rec->t = rt;
		nor = (r.origin + p * rt - center) / radius;
		if (nor.dot(p) > 0) {
			rec->n = -nor;
		}
		else {
			rec->n = nor;
		}
		return dsc > 0;
	}

	point center;
	float radius;
};

class camera {
public:
	camera() = default;
	camera(point _look_from, point _look_at, point _up, double _vfov, double _aspect_ratio, double _focus) :
		look_from(_look_from),
		look_at(_look_at),
		up(_up),
		vfov(_vfov),
		aspect_ratio(_aspect_ratio),
		focus(_focus) {

		w = (look_from - look_at).normalized();
		u = up.cross(w).normalized();
		v = w.cross(u).normalized();

		double width, height;
		/*height = focus * tan(vfov / 2);
		width = height * aspect_ratio;*/
		width = 2.82;
		height = 1.58;
		u = u * width;
		v = v * height;
		up_left_corner = look_from - w * focus + v / 2 - u / 2;
	}
	ray get_ray(double h, double vt) {
		point emit = up_left_corner + h * u - vt * v;
		return ray(look_from, emit - look_from);
	}
	vec3f u, v, w;
	point up_left_corner;
	point look_from;
	point look_at;
	point up;
	double vfov;
	double aspect_ratio;
	double focus;
};
const unsigned int width_sampling = 1920;
const unsigned int height_sampling = 1080;
const unsigned int multi_sampling = 3000;

//const float view_port_width = 16;
//const float view_port_height = 9;
//const float view_focus_length = 4.5;
//
//vec3f view_port_horizontal	(view_port_width, 0, 0);
//vec3f view_port_vertical	(0, view_port_height, 0);
//vec3f view_focus_vec		(0, 0, -view_focus_length);
//point camera_position		(0, 0, 0);
//point up_left_corner = camera_position + view_focus_vec - view_port_horizontal / 2 + view_port_vertical / 2;

std::vector<hittable*> world;
camera c(point(-2, 2, 1), point(0, 0, -1), vec3f(0, 1, 0), 3.14 / 2, float(16) / 9, 4.5);

color get_color(ray r, int dep = 50) {
	if (dep >= 0) {
		hit_record rec;
		double far = 1e50;
		hit_record get_rec;
		bool hit = false;
		int ind;
		color attenuation;
		ray scattered;
		for (int i = 0; i < world.size(); ++i) {
			if (world[i]->hit(r, &rec)) {
				hit = true;
				if (rec.t < far) {
					far = rec.t;
					ind = i;
					get_rec = rec;
				}
			}
		}
		if (hit) {
			world[ind]->m->scattar(r, get_rec, &attenuation, &scattered);
			// 非发光体
			if (scattered.dir != vec3f(0, 0, 0)) {
				color res = get_color(scattered, dep - 1);
				return color(attenuation.x() * res.x(), attenuation.y() * res.y(), attenuation.z() * res.z());
			}
			// 发光体返回(0, 0, 0)光线方向
			else {
				return color(attenuation.x(), attenuation.y(), attenuation.z());
			}
		}
		vec3f v = r.dir.normalized();
		float t = 0.5 * (v.y() + 1.0f);

		// return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
		return color(0, 0, 0);
	}
	return color(0, 0, 0);
}

int main() {
	material* m1 = new metal(color(0.8, 0.8, 0.8));
	material* m2 = new metal(color(.8, .6, .2));
	material* m3 = new lambertian(color(.8, .8, .8));
	material* m4 = new lambertian(color(.7, .3, .3));
	material* l = new luminous(color(1, .9, .9));

	world.push_back(new sphere(vec3f(-1, 0, -1), 0.5, m1));
	world.push_back(new sphere(vec3f(0, 0, -1), 0.5, l));
	world.push_back(new sphere(vec3f(1, 0, -1), 0.5, m2));

	world.push_back(new sphere(vec3f(0, -100.5, -1), 100, m3));

	data = (pixel*)malloc(sizeof(pixel) * 1920 * 1080);
	omp_set_num_threads(16);
	#pragma omp parallel for
	for (int y = height_sampling - 1; y >= 0; --y) {
		std::cout << y << std::endl;
		#pragma omp parallel for
		for (int x = 0; x < width_sampling; ++x) {
			color col;
			#pragma omp parallel for
			for (int k = 0; k < multi_sampling; ++k) {
				float h_bias = u(e), v_bias = u(e);
				/*vec3f vertical_bias = (y + v_bias) / (height_sampling) * (-view_port_vertical);
				vec3f horizontal_bias = (x + h_bias) / (width_sampling) * view_port_horizontal;
				point emit_point = up_left_corner + vertical_bias + horizontal_bias;
				ray r(camera_position, emit_point - camera_position);*/
				double h = (x + h_bias) / (width_sampling);
				double vt = (y + v_bias) / (height_sampling);
				ray r = c.get_ray(h, vt);
				if (k == 0)
					col = get_color(r);
				else
					col += get_color(r);
			}
			col /= multi_sampling;
			data[y * 1920 + x].r = col.x() * 255;
			data[y * 1920 + x].g = col.y() * 255;
			data[y * 1920 + x].b = col.z() * 255;
		}
	}
	stbi_write_jpg("../render_result/render.jpg", 1920, 1080, 3, data, 100);

	for (std::vector<hittable*>::iterator it = world.begin();
		it != world.end();
		++it) {
		free(*it);
	}
	delete(m1);
	delete(m2);
	delete(m3);
	delete(m4);

	free(data);
	return 0;
}
