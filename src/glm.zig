const std = @import("std");
const sqrt = std.math.sqrt;
const cos = std.math.cos;
const sin = std.math.sin;
const tan = std.math.tan;
const approxEqAbs = std.math.approxEqAbs;

pub fn clamp(v: f32, lo: f32, hi: f32) f32 {
    return @max(@min(v, hi), lo);
}

pub fn radians(angle: f32) f32 {
    return angle * (3.1415 / 180.0);
}

// TODO: Decide on whether all the functions
// should return Self or not, currently the only
// function that manipulates itself is normalize
fn Vector(comptime dim: usize) type {
    // Extern due to being compatible with C ABI for OpenGL
    return extern struct {
        vals: @Vector(dim, f32),

        const Self = @This();

        // Constructors
        pub fn fill(val: f32) Self {
            const vals = @as(@Vector(dim, f32), @splat(val));
            return Self{ .vals = vals };
        }

        pub fn zeros() Self {
            return Self.fill(0.0);
        }

        pub fn ones() Self {
            return Self.fill(1.0);
        }

        pub fn sum(self: Self) f32 {
            var total: f32 = 0.0;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                total += self.vals[i];
            }
            return total;
        }

        pub fn add(self: Self, other: Self) Self {
            return Self{ .vals = self.vals + other.vals };
        }

        pub fn addScalar(self: Self, scalar: f32) Self {
            const vals_scalar = @as(@Vector(dim, f32), @splat(scalar));
            return Self{ .vals = self.vals + vals_scalar };
        }

        pub fn sub(self: Self, other: Self) Self {
            return Self{ .vals = self.vals - other.vals };
        }

        pub fn subScalar(self: Self, scalar: f32) Self {
            const vals_scalar = @as(@Vector(dim, f32), @splat(scalar));
            return Self{ .vals = self.vals - vals_scalar };
        }

        pub fn mul(self: Self, other: Self) Self {
            return Self{ .vals = self.vals * other.vals };
        }

        pub fn mulScalar(self: Self, scalar: f32) Self {
            const vals_scalar = @as(@Vector(dim, f32), @splat(scalar));
            return Self{ .vals = self.vals * vals_scalar };
        }

        pub fn div(self: Self, other: Self) Self {
            return Self{ .vals = self.vals / other.vals };
        }

        pub fn divScalar(self: Self, scalar: f32) Self {
            const vals_scalar = @as(@Vector(dim, f32), @splat(scalar));
            return Self{ .vals = self.vals / vals_scalar };
        }

        pub fn neg(self: Self) Self {
            const zero_vals = @as(@Vector(dim, f32), @splat(0.0));
            return Self{ .vals = zero_vals - self.vals };
        }

        pub fn dot(self: Self, other: Self) f32 {
            const product = self.mul(other);
            return product.sum();
        }

        pub fn magnitudeSq(self: Self) f32 {
            return self.dot(self);
        }

        pub fn magnitude(self: Self) f32 {
            return sqrt(self.magnitudeSq());
        }

        pub fn normalize(self: Self) Self {
            const n = self.magnitude();
            const vals = self.vals / @as(@Vector(dim, f32), @splat(n));
            return Self{ .vals = vals };
        }

        pub fn cross(self: Self, other: Self) Self {
            if (dim != 3) {
                @compileError("Cross product only defined for 3D vectors");
            }
            const vals = [3]f32{
                self.vals[1] * other.vals[2] - self.vals[2] * other.vals[1],
                self.vals[2] * other.vals[0] - self.vals[0] * other.vals[2],
                self.vals[0] * other.vals[1] - self.vals[1] * other.vals[0],
            };
            return Self{ .vals = vals };
        }

        pub fn compare(self: Self, other: Self, tolerance: f32) bool {
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                if (!approxEqAbs(f32, self.vals[i], other.vals[i], tolerance)) {
                    return false;
                }
            }
            return true;
        }
    };
}

pub const Vec2 = Vector(2);
pub const Vec3 = Vector(3);
pub const Vec4 = Vector(4);

pub fn vec2(x: f32, y: f32) Vec2 {
    return Vec2{ .vals = [2]f32{ x, y } };
}

pub fn vec3(x: f32, y: f32, z: f32) Vec3 {
    return Vec3{ .vals = [3]f32{ x, y, z } };
}

pub fn vec4(x: f32, y: f32, z: f32, w: f32) Vec4 {
    return Vec4{ .vals = [4]f32{ x, y, z, w } };
}

fn Matrix(comptime dim: usize) type {
    return extern struct {
        /// The internal Matrix values represented as `dim` number of Zig @Vector's
        /// of length `dim`.
        vals: [dim]@Vector(dim, f32),

        const Self = @This();

        /// Creates matrix of dimensions `dim` x `dim` and fills it with zeroes.
        pub fn zeros() Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    vals[i][j] = 0.0;
                }
            }
            return Self{ .vals = vals };
        }

        /// Creates identity matrix of dimensions `dim` x `dim`
        pub fn identity() Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    vals[i][j] = if (i == j) 1.0 else 0.0;
                }
            }
            return Self{ .vals = vals };
        }

        /// Returns the transpose of the given matrix
        pub fn transpose(self: Self) Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    vals[i][j] = self.vals[j][i];
                }
            }
            return Self{ .vals = vals };
        }

        /// Matrix addition between matrix `self` and matrix `other`, returns resulting matrix.
        pub fn matAdd(self: Self, other: Self) Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                const self_row = self.vals[i];
                const other_row = other.vals[i];
                vals[i] = self_row + other_row;
            }
            return Self{ .vals = vals };
        }

        /// Matrix subtraction between matrix `self` and `other`, returns resulting matrix.
        pub fn matSub(self: Self, other: Self) Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                const self_row = self.vals[i];
                const other_row = other.vals[i];
                vals[i] = self_row - other_row;
            }
            return Self{ .vals = vals };
        }

        /// Matrix scalar multiplication, multiplies matrix `self` with `scalar`, returns resulting matrix.
        pub fn matMulScalar(self: Self, scalar: f32) Self {
            var vals: [dim]@Vector(dim, f32) = undefined;
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                const self_row = self.vals[i];
                const scalar_row = @as(@Vector(dim, f32), @splat(scalar));
                vals[i] = self_row * scalar_row;
            }
            return Self{ .vals = vals };
        }

        /// Matrix-Matrix multiplication, multiplies matrix `self` with matrix `other`, returns resulting matrix.
        /// Its important to note that this only works for square matrices currently, meaning `self` and `other`
        /// must have the same dimensions.
        pub fn matMul(self: Self, other: Self) Self {
            // NOTE: This works because its a square matrix,
            // so the MxN mul NxM condition is satisified
            var vals: [dim]@Vector(dim, f32) = undefined;
            const a = self;
            const b = other.transpose();

            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    const row = a.vals[i];
                    const col = b.vals[j];
                    const prod: [dim]f32 = row * col;

                    var sum: f32 = 0;
                    for (prod) |p| {
                        sum += p;
                    }
                    vals[i][j] = sum;
                }
            }
            return Self{ .vals = vals };
        }

        /// Compare equality between two matrices `self` and `other` using a tolerance value `tolerance`
        pub fn compare(self: Self, other: Self, tolerance: f32) bool {
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    if (!approxEqAbs(f32, self.vals[i][j], other.vals[i][j], tolerance)) {
                        return false;
                    }
                }
            }
            return true;
        }

        /// Debug print of the matrix
        pub fn print(self: Self) void {
            std.debug.print("Mat: ", .{});
            comptime var i = 0;
            inline while (i < dim) : (i += 1) {
                comptime var j = 0;
                inline while (j < dim) : (j += 1) {
                    std.debug.print("{} ", .{self.vals[i][j]});
                }
                std.debug.print("\n", .{});
            }
        }
    };
}

/// Two-dimensional matrix
pub const Mat2 = Matrix(2);
/// Three-dimensional matrix
pub const Mat3 = Matrix(3);
/// Four-dimensional matrix
pub const Mat4 = Matrix(4);

// TODO (Thomas): There's probably a way to do this with comptime
// as a member function inside the Matrix struct instead of duplicating
// the function for the different matrix-vector dimensions like its done here.
pub fn mat2MulVec2(mat: Mat2, vec: Vec2) Vec2 {
    const dim: usize = 2;
    var result_vec = Vector(2).fill(0.0);

    comptime var i: usize = 0;
    inline while (i < dim) : (i += 1) {
        const row: @Vector(dim, f32) = mat.vals[i];
        const col = vec.vals;

        const prod: [dim]f32 = row * col;

        var sum: f32 = 0;
        for (prod) |p| {
            sum += p;
        }
        result_vec.vals[i] = sum;
    }

    return result_vec;
}

pub fn mat3MulVec3(mat: Mat3, vec: Vec3) Vec3 {
    const dim: usize = 3;
    var result_vec = Vector(3).fill(0.0);

    comptime var i: usize = 0;
    inline while (i < dim) : (i += 1) {
        const row: @Vector(dim, f32) = mat.vals[i];
        const col = vec.vals;

        const prod: [dim]f32 = row * col;

        var sum: f32 = 0;
        for (prod) |p| {
            sum += p;
        }
        result_vec.vals[i] = sum;
    }

    return result_vec;
}

pub fn mat4MulVec4(mat: Mat4, vec: Vec4) Vec4 {
    const dim: usize = 4;
    var result_vec = Vector(4).fill(0.0);

    comptime var i: usize = 0;
    inline while (i < dim) : (i += 1) {
        const row: @Vector(dim, f32) = mat.vals[i];
        const col = vec.vals;

        const prod: [dim]f32 = row * col;

        var sum: f32 = 0;
        for (prod) |p| {
            sum += p;
        }
        result_vec.vals[i] = sum;
    }

    return result_vec;
}

/// Transformation matrix for scale by 3-dimensional vector 'vec'
pub fn scaleMatrix(vec: Vec3) Mat4 {
    return Mat4{
        .vals = [4]@Vector(4, f32){
            .{ vec.vals[0], 0.0, 0.0, 0.0 },
            .{ 0.0, vec.vals[1], 0.0, 0.0 },
            .{ 0.0, 0.0, vec.vals[2], 0.0 },
            .{ 0.0, 0.0, 0.0, 1.0 },
        },
    };
}

pub fn scale(mat: Mat4, vec: Vec3) Mat4 {
    const scale_matrix = scaleMatrix(vec);
    return mat.matMul(scale_matrix);
}

/// Transformation matrix for translation by vec
pub fn translationMatrix(vec: Vec3) Mat4 {
    return Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, vec.vals[0] },
        .{ 0.0, 1.0, 0.0, vec.vals[1] },
        .{ 0.0, 0.0, 1.0, vec.vals[2] },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
}

pub fn translate(mat: Mat4, vec: Vec3) Mat4 {
    const translation_matrix = translationMatrix(vec);
    return mat.matMul(translation_matrix);
}

pub fn rotationX(angle: f32) Mat4 {
    return Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, cos(angle), -sin(angle), 0.0 },
        .{ 0.0, sin(angle), cos(angle), 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
}

pub fn rotationY(angle: f32) Mat4 {
    return Mat4{ .vals = [4]@Vector(4, f32){
        .{ cos(angle), 0.0, sin(angle), 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ -sin(angle), 0.0, cos(angle), 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
}

pub fn rotationZ(angle: f32) Mat4 {
    return Mat4{ .vals = [4]@Vector(4, f32){
        .{ cos(angle), -sin(angle), 0.0, 0.0 },
        .{ sin(angle), cos(angle), 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
}

/// Angle in degrees
pub fn rotationMatrix(angle: f32, axis: Vec3) Mat4 {
    const unit = axis.normalize();
    const x = unit.vals[0];
    const y = unit.vals[1];
    const z = unit.vals[2];

    const a = cos(angle) + x * x * (1 - cos(angle));
    const b = x * y * (1 - cos(angle)) - z * sin(angle);
    const c = x * z * (1 - cos(angle)) + y * sin(angle);
    const d = y * x * (1 - cos(angle)) + z * sin(angle);
    const e = cos(angle) + y * y * (1 - cos(angle));
    const f = y * z * (1 - cos(angle)) - x * sin(angle);
    const h = z * x * (1 - cos(angle)) - y * sin(angle);
    const i = z * y * (1 - cos(angle)) + x * sin(angle);
    const j = cos(angle) + z * z * (1 - cos(angle));

    return Mat4{ .vals = [4]@Vector(4, f32){
        .{ a, b, c, 0.0 },
        .{ d, e, f, 0.0 },
        .{ h, i, j, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
}

pub fn rotate(mat: Mat4, angle: f32, vec: Vec3) Mat4 {
    const rotation_matrix = rotationMatrix(angle, vec);
    return mat.matMul(rotation_matrix);
}

// TODO: This needs to be thouroghly verified
pub fn lookAt(pos: Vec3, target: Vec3, up: Vec3) Mat4 {
    var direction = pos.sub(target).normalize();
    const cam_right = up.cross(direction).normalize();
    const cam_up = direction.cross(cam_right);

    var axes = Mat4{ .vals = [4]@Vector(4, f32){
        .{ cam_right.vals[0], cam_right.vals[1], cam_right.vals[2], 0.0 },
        .{ cam_up.vals[0], cam_up.vals[1], cam_up.vals[2], 0.0 },
        .{ direction.vals[0], direction.vals[1], direction.vals[2], 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };

    const pos_matrix = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, -pos.vals[0] },
        .{ 0.0, 1.0, 0.0, -pos.vals[1] },
        .{ 0.0, 0.0, 1.0, -pos.vals[2] },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };

    return axes.matMul(pos_matrix);
}

/// Perspective projection matrix
pub fn perspective(fov: f32, aspect: f32, znear: f32, zfar: f32) Mat4 {
    const tan_half_fov = tan(fov / 2.0);
    const a = 1.0 / (aspect * tan_half_fov);
    const b = 1.0 / tan_half_fov;
    const c = -(zfar + znear) / (zfar - znear);
    const d = -(2.0 * zfar * znear) / (zfar - znear);

    return Mat4{
        .vals = [4]@Vector(4, f32){
            .{ a, 0.0, 0.0, 0.0 },
            .{ 0.0, b, 0.0, 0.0 },
            .{ 0.0, 0.0, c, d },
            .{ 0.0, 0.0, -1.0, 0.0 },
        },
    };
}

/// Orthographic projection matrix
pub fn ortho(left: f32, right: f32, bottom: f32, top: f32, znear: f32, zfar: f32) Mat4 {
    const a = 2.0 / (right - left);
    const b = 2.0 / (top - bottom);
    const c = -2.0 / (zfar - znear);
    const d = -(right + left) / (right - left);
    const e = -(top + bottom) / (top - bottom);
    const f = -(zfar + znear) / (zfar - znear);

    return Mat4{
        .vals = [4]@Vector(4, f32){
            .{ a, 0.0, 0.0, d },
            .{ 0.0, b, 0.0, e },
            .{ 0.0, 0.0, c, f },
            .{ 0.0, 0.0, 0.0, 0.0 },
        },
    };
}

const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

test "clamp" {
    try expectEqual(@as(f32, 5.0), clamp(10.0, 0.0, 5.0)); // clamp high value
    try expectEqual(@as(f32, 0.0), clamp(-5.0, 0.0, 10.0)); // clamp low value
    try expectEqual(@as(f32, 5.0), clamp(5.0, 0.0, 10.0)); // value in range
}

test "Vector.fill" {
    var actual: Vec4 = Vec4.fill(1.0);
    const expected: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    try expect(actual.compare(expected, 0.00001));
}

test "Vector.zeros" {
    var actual: Vec4 = Vec4.zeros();
    const expected: Vec4 = vec4(0.0, 0.0, 0.0, 0.0);
    try expect(actual.compare(expected, 0.00001));
}

test "Vector.ones" {
    var actual: Vec4 = Vec4.ones();
    const expected: Vec4 = vec4(1.0, 1.0, 1.0, 1.0);
    try expect(actual.compare(expected, 0.00001));
}

test "Vector.sum" {
    var actual: Vec4 = Vec4{ .vals = [4]f32{ 1.0, 2.0, 3.0, 4.0 } };
    try expectEqual(@as(f32, 10.0), actual.sum());
}

test "Vector.add" {
    var vec_a: Vec4 = Vec4.ones();
    const vec_b: Vec4 = Vec4.ones();
    var result: Vec4 = vec_a.add(vec_b);
    const expected = vec4(2.0, 2.0, 2.0, 2.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.addScalar" {
    var vec_a: Vec4 = Vec4.ones();
    const scalar: f32 = 1.0;
    var result: Vec4 = vec_a.addScalar(scalar);
    const expected = vec4(2.0, 2.0, 2.0, 2.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.sub" {
    var vec_a: Vec4 = vec4(2.0, 2.0, 2.0, 2.0);
    const vec_b: Vec4 = Vec4.ones();
    var result = vec_a.sub(vec_b);
    const expected = vec4(1.0, 1.0, 1.0, 1.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.subScalar" {
    var vec_a: Vec4 = vec4(2.0, 2.0, 2.0, 2.0);
    const scalar: f32 = 2.0;
    var result = vec_a.subScalar(scalar);
    const expected = vec4(0.0, 0.0, 0.0, 0.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.mul" {
    var vec_a: Vec4 = vec4(2.0, 2.0, 2.0, 2.0);
    const vec_b: Vec4 = vec4(3.0, 3.0, 3.0, 3.0);
    var result: Vec4 = vec_a.mul(vec_b);
    const expected = vec4(6.0, 6.0, 6.0, 6.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.mulScalar" {
    var vec_a: Vec4 = vec4(2.0, 2.0, 2.0, 2.0);
    const scalar: f32 = 2.0;
    var result: Vec4 = vec_a.mulScalar(scalar);
    const expected = vec4(4.0, 4.0, 4.0, 4.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.div" {
    var vec_a: Vec4 = vec4(6.0, 6.0, 6.0, 6.0);
    const vec_b: Vec4 = vec4(2.0, 2.0, 2.0, 2.0);
    var result: Vec4 = vec_a.div(vec_b);
    const expected = vec4(3.0, 3.0, 3.0, 3.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.divScalar" {
    var vec_a: Vec4 = vec4(6.0, 6.0, 6.0, 6.0);
    const scalar: f32 = 2.0;
    var result: Vec4 = vec_a.divScalar(scalar);
    const expected = vec4(3.0, 3.0, 3.0, 3.0);
    try expect(result.compare(expected, 0.00001));
}

test "Vector.dot" {
    var vec_a: Vec3 = vec3(1.0, 2.0, 3.0);
    const vec_b: Vec3 = vec3(4.0, 5.0, 6.0);
    const result: f32 = vec_a.dot(vec_b);
    const expected: f32 = 32.0;
    try expect(approxEqAbs(f32, expected, result, 0.00001));
}

test "Vector.magnitude" {
    var vec: Vec3 = vec3(1.0, 2.0, 2.0);
    const result: f32 = vec.magnitude();
    const expected: f32 = 3.0;
    try expect(approxEqAbs(f32, expected, result, 0.00001));
}

test "Vector.normalize" {
    var vec: Vec3 = vec3(1.0, 2.0, 3.0);
    var result: Vec3 = vec.normalize();
    const expected_mag: f32 = 1.0;
    try expect(approxEqAbs(f32, expected_mag, result.magnitude(), 0.00001));
}

test "Vector.cross" {
    var vec_a: Vec3 = vec3(1.0, 2.0, 3.0);
    const vec_b: Vec3 = vec3(4.0, 5.0, 6.0);
    var result: Vec3 = vec_a.cross(vec_b);
    const expected: Vec3 = vec3(-3.0, 6.0, -3.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 identity" {
    var mat = Mat2.identity();
    const expected: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix2 zeros" {
    var mat = Mat2.zeros();
    const expected: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 0.0, 0.0 },
        .{ 0.0, 0.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix2 transpose" {
    var mat: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    } };
    var result: Mat2 = mat.transpose();
    const expected: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 3.0 },
        .{ 2.0, 4.0 },
    } };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matAdd" {
    var mat_a: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    } };
    const mat_b: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 5.0, 6.0 },
            .{ 7.0, 8.0 },
        },
    };
    var result = mat_a.matAdd(mat_b);
    const expected: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 6.0, 8.0 },
            .{ 10.0, 12.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matSub" {
    var mat_a: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 4.0, 2.0 },
        .{ 1.0, 6.0 },
    } };
    const mat_b: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 2.0, 4.0 },
            .{ 0.0, 1.0 },
        },
    };
    var result = mat_a.matSub(mat_b);
    const expected: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 2.0, -2.0 },
            .{ 1.0, 5.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matMulScalar" {
    var mat_a: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    } };
    const scalar: f32 = 2.0;
    var result = mat_a.matMulScalar(scalar);

    const expected: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 2.0, 4.0 },
            .{ 6.0, 8.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matMul" {
    var mat_a: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    } };
    const mat_b: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 5.0, 6.0 },
            .{ 7.0, 8.0 },
        },
    };
    var result = mat_a.matMul(mat_b);

    const expected: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 19.0, 22.0 },
            .{ 43.0, 50.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matMul 2" {
    var mat_a: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 4.0 },
    } };
    const mat_b: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 1.0, 2.0 },
            .{ 3.0, 4.0 },
        },
    };
    var result = mat_a.matMul(mat_b);

    const expected: Mat2 = Mat2{
        .vals = [2]@Vector(2, f32){
            .{ 7.0, 10.0 },
            .{ 15.0, 22.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix2 matMulVec identity" {
    const mat: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 0.0 },
        .{ 0.0, 1.0 },
    } };
    const vec: Vec2 = vec2(1.0, 2.0);
    var result: Vec2 = mat2MulVec2(mat, vec);

    const expected: Vec2 = vec2(1.0, 2.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix matMulVec 1" {
    const mat: Mat2 = Mat2{ .vals = [2]@Vector(2, f32){
        .{ 1.0, 2.0 },
        .{ 3.0, 2.0 },
    } };
    const vec: Vec2 = vec2(5.0, 6.0);
    var result: Vec2 = mat2MulVec2(mat, vec);

    const expected: Vec2 = vec2(17.0, 27.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 identity" {
    var mat = Mat3.identity();
    const expected: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix3 zeros" {
    var mat = Mat3.zeros();
    const expected: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix3 transpose" {
    var mat: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
        .{ 7.0, 8.0, 9.0 },
    } };
    var result = mat.transpose();
    const expected: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 4.0, 7.0 },
        .{ 2.0, 5.0, 8.0 },
        .{ 3.0, 6.0, 9.0 },
    } };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matAdd" {
    var mat_a: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
        .{ 7.0, 8.0, 9.0 },
    } };

    const mat_b: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 10.0, 11.0, 12.0 },
            .{ 13.0, 14.0, 15.0 },
            .{ 16.0, 17.0, 18.0 },
        },
    };
    var result = mat_a.matAdd(mat_b);

    const expected: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 11.0, 13.0, 15.0 },
            .{ 17.0, 19.0, 21.0 },
            .{ 23.0, 25.0, 27.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matSub" {
    var mat_a: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
        .{ 7.0, 8.0, 9.0 },
    } };

    const mat_b: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 10.0, 11.0, 12.0 },
            .{ 13.0, 14.0, 15.0 },
            .{ 16.0, 17.0, 18.0 },
        },
    };
    var result = mat_a.matSub(mat_b);

    const expected: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ -9.0, -9.0, -9.0 },
            .{ -9.0, -9.0, -9.0 },
            .{ -9.0, -9.0, -9.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matMulScalar" {
    var mat_a: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
        .{ 7.0, 8.0, 9.0 },
    } };

    const scalar: f32 = 2.0;
    var result = mat_a.matMulScalar(scalar);

    const expected: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 2.0, 4.0, 6.0 },
            .{ 8.0, 10.0, 12.0 },
            .{ 14.0, 16.0, 18.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matMul" {
    var mat_a: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 4.0, 5.0, 6.0 },
        .{ 7.0, 8.0, 9.0 },
    } };

    const mat_b: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 1.0, 2.0, 3.0 },
            .{ 4.0, 5.0, 6.0 },
            .{ 7.0, 8.0, 9.0 },
        },
    };
    var result = mat_a.matMul(mat_b);

    const expected: Mat3 = Mat3{
        .vals = [3]@Vector(3, f32){
            .{ 30.0, 36.0, 42.0 },
            .{ 66.0, 81.0, 96.0 },
            .{ 102.0, 126.0, 150.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matMulVec identity" {
    const mat: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
    } };
    const vec: Vec3 = vec3(1.0, 2.0, 3.0);
    var result: Vec3 = mat3MulVec3(mat, vec);

    const expected: Vec3 = vec3(1.0, 2.0, 3.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix3 matMulVec 1" {
    const mat: Mat3 = Mat3{ .vals = [3]@Vector(3, f32){
        .{ 1.0, 2.0, 3.0 },
        .{ 3.0, 2.0, 1.0 },
        .{ 1.0, 2.0, 3.0 },
    } };
    const vec: Vec3 = vec3(4.0, 5.0, 6.0);
    var result: Vec3 = mat3MulVec3(mat, vec);

    const expected: Vec3 = vec3(32.0, 28.0, 32.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 identity" {
    var mat = Mat4.identity();
    const expected: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix4 zeros" {
    var mat = Mat4.zeros();
    const expected: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 0.0 },
    } };
    try expect(mat.compare(expected, 0.00001));
}

test "Matrix4 transpose" {
    var mat: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 2.0, 3.0, 4.0 },
        .{ 5.0, 6.0, 7.0, 8.0 },
        .{ 9.0, 10.0, 11.0, 12.0 },
        .{ 13.0, 14.0, 15.0, 16.0 },
    } };
    var result = mat.transpose();
    const expected: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 5.0, 9.0, 13.0 },
        .{ 2.0, 6.0, 10.0, 14.0 },
        .{ 3.0, 7.0, 11.0, 15.0 },
        .{ 4.0, 8.0, 12.0, 16.0 },
    } };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matAdd" {
    var mat_a: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 2.0, 3.0, 4.0 },
        .{ 5.0, 6.0, 7.0, 8.0 },
        .{ 9.0, 10.0, 11.0, 12.0 },
        .{ 13.0, 14.0, 15.0, 16.0 },
    } };

    const mat_b: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ 1.0, 2.0, 3.0, 4.0 },
            .{ 5.0, 6.0, 7.0, 8.0 },
            .{ 9.0, 10.0, 11.0, 12.0 },
            .{ 13.0, 14.0, 15.0, 16.0 },
        },
    };
    var result = mat_a.matAdd(mat_b);

    const expected: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ 2.0, 4.0, 6.0, 8.0 },
            .{ 10.0, 12.0, 14.0, 16.0 },
            .{ 18.0, 20.0, 22.0, 24.0 },
            .{ 26.0, 28.0, 30.0, 32.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matSub" {
    var mat_a: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ 1.0, 2.0, 3.0, 4.0 },
            .{ 5.0, 6.0, 7.0, 8.0 },
            .{ 9.0, 10.0, 11.0, 12.0 },
            .{ 13.0, 14.0, 15.0, 16.0 },
        },
    };

    const mat_b: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 2.0, 3.0, 4.0 },
        .{ 5.0, 6.0, 7.0, 8.0 },
        .{ 9.0, 10.0, 11.0, 12.0 },
        .{ 13.0, 14.0, 15.0, 16.0 },
    } };

    var result = mat_a.matSub(mat_b);

    const expected: Mat4 = Mat4.zeros();
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matMulScalar" {
    var mat_a: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 2.0, 3.0, 4.0 },
        .{ 5.0, 6.0, 7.0, 8.0 },
        .{ 9.0, 10.0, 11.0, 12.0 },
        .{ 13.0, 14.0, 15.0, 16.0 },
    } };

    const scalar: f32 = -2.0;
    var result = mat_a.matMulScalar(scalar);

    const expected: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ -2.0, -4.0, -6.0, -8.0 },
            .{ -10.0, -12.0, -14.0, -16.0 },
            .{ -18.0, -20.0, -22.0, -24.0 },
            .{ -26.0, -28.0, -30.0, -32.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matMul" {
    var mat_a: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 2.0, 3.0, 4.0 },
        .{ 5.0, 6.0, 7.0, 8.0 },
        .{ 9.0, 10.0, 11.0, 12.0 },
        .{ 13.0, 14.0, 15.0, 16.0 },
    } };

    const mat_b: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ 17.0, 3.0, 1.0, -4.0 },
            .{ 2.0, 6.0, 9.0, 8.0 },
            .{ 9.0, 22.0, 14.0, 12.0 },
            .{ 13.0, 78.0, 15.0, 10.0 },
        },
    };
    var result = mat_a.matMul(mat_b);

    const expected: Mat4 = Mat4{
        .vals = [4]@Vector(4, f32){
            .{ 100.0, 393.0, 121.0, 88.0 },
            .{ 264.0, 829.0, 277.0, 192.0 },
            .{ 428.0, 1265.0, 433.0, 296.0 },
            .{ 592.0, 1701.0, 589.0, 400.0 },
        },
    };
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matMulVec identity" {
    const mat: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
    const vec: Vec4 = vec4(1.0, 2.0, 3.0, 4.0);
    var result: Vec4 = mat4MulVec4(mat, vec);

    const expected: Vec4 = vec4(1.0, 2.0, 3.0, 4.0);
    try expect(result.compare(expected, 0.00001));
}

test "Matrix4 matMulVec 1" {
    const mat: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 11.0, 26.0, -3.0, 72.0 },
        .{ 0.0, 234.0, 11.0, 69.0 },
        .{ 13.0, 23.0, -34.0, 2.0 },
        .{ -1.0, 13.0, -42.0, 69.0 },
    } };
    const vec: Vec4 = vec4(0.0, 6.0, -4.0, 11.0);
    var result: Vec4 = mat4MulVec4(mat, vec);

    const expected: Vec4 = vec4(960.0, 2119.0, 296.0, 1005.0);
    try expect(result.compare(expected, 0.00001));
}

test "Scale Matrix" {
    const scale_vec = vec3(2.0, 2.0, 2.0);
    var scale_mat = scaleMatrix(scale_vec);
    const expected_scale_mat: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 2.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 2.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 2.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };

    try expect(scale_mat.compare(expected_scale_mat, 0.00001));

    const original_vec = vec4(1.0, 3.0, 7.0, 1.0);

    var scaled_vec = mat4MulVec4(scale_mat, original_vec);
    const expected_scaled_vec = vec4(2.0, 6.0, 14.0, 1.0);

    try expect(scaled_vec.compare(expected_scaled_vec, 0.00001));
}

test "Translate Matrix" {
    const translate_vec = vec3(2.0, 3.0, -1.0);
    var translate_mat = translationMatrix(translate_vec);
    const expected_translate_mat: Mat4 = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 2.0 },
        .{ 0.0, 1.0, 0.0, 3.0 },
        .{ 0.0, 0.0, 1.0, -1.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };

    try expect(translate_mat.compare(expected_translate_mat, 0.00001));

    const original_vec = vec4(2.0, 1.0, 5.0, 1.0);

    var translated_vec = mat4MulVec4(translate_mat, original_vec);
    const expected_translated_vec = vec4(4.0, 4.0, 4.0, 1.0);

    try expect(translated_vec.compare(expected_translated_vec, 0.00001));
}

test "Rotation Matrix Zero Degree Angle" {
    const axis = vec3(0.0, 1.0, 0.0);
    const angle: f32 = 0.0;
    var rotation_mat = rotationMatrix(angle, axis);
    const expected_rotation_mat = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };
    try expect(rotation_mat.compare(expected_rotation_mat, 0.00001));

    const original_vec = vec4(3.0, 1.0, 2.5, 1.0);
    var rotated_vec = mat4MulVec4(rotation_mat, original_vec);
    const expected_rotated_vec = vec4(3.0, 1.0, 2.5, 1.0);

    try expect(rotated_vec.compare(expected_rotated_vec, 0.00001));
}

test "Rotation Matrix X 90 Degree Angle" {
    const axis = vec3(1.0, 0.0, 0.0);
    const angle: f32 = 90.0;
    var rotation_mat = rotationMatrix(angle, axis);

    const expected_rotation_mat = Mat4{ .vals = [4]@Vector(4, f32){
        .{ 1.0, 0.0, 0.0, 0.0 },
        .{ 0.0, cos(angle), -sin(angle), 0.0 },
        .{ 0.0, sin(angle), cos(angle), 0.0 },
        .{ 0.0, 0.0, 0.0, 1.0 },
    } };

    try expect(rotation_mat.compare(expected_rotation_mat, 0.00001));

    const original_point = vec4(2.0, 3.0, 1.0, 1.0);
    var rotated_point = mat4MulVec4(rotation_mat, original_point);

    const expected_rotated_point = vec4(
        original_point.vals[0],
        cos(angle) * original_point.vals[1] - sin(angle) * original_point.vals[2],
        sin(angle) * original_point.vals[1] + cos(angle) * original_point.vals[2],
        1.0,
    );
    try expect(rotated_point.compare(expected_rotated_point, 0.00001));
}
