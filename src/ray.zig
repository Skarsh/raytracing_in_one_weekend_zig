const glm = @import("glm.zig");

pub const Ray = struct {
    orig: glm.Vec3,
    dir: glm.Vec3,

    pub fn init(orig: glm.Vec3, dir: glm.Vec3) Ray {
        return Ray{ .orig = orig, .dir = dir };
    }

    pub fn origin(self: Ray) glm.Vec3 {
        return self.orig;
    }

    pub fn direction(self: Ray) glm.Vec3 {
        return self.dir;
    }

    pub fn at(self: Ray, t: f32) glm.Vec3 {
        return self.orig + (t * self.dir);
    }
};
