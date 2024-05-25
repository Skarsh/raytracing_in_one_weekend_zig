const std = @import("std");

const color = @import("color.zig");
const glm = @import("glm.zig");
const ray = @import("ray.zig");

fn rayColor(r: ray.Ray) color.Color {
    const unit_direction = r.direction().normalize();
    const a = 0.5 * (unit_direction.vals[1] + 1.0);
    return glm.Vec3.ones().mulScalar(1.0 - a).add(glm.vec3(0.5, 0.7, 1.0).mulScalar(a));
}

pub fn main() !void {

    // Image
    const aspect_ratio: f32 = 16.0 / 9.0;
    const image_width: u32 = 400;

    // Calculate the image_heigth, and ensure it's at least 1
    var image_height: u32 = image_width / aspect_ratio;
    image_height = if (image_height < 1) 1 else image_height;

    // Camera
    const focal_length: f32 = 1.0;
    const viewport_height: f32 = 2.0;
    const viewport_width: f32 = viewport_height * (@as(f32, @floatFromInt(image_width)) / @as(f32, @floatFromInt(image_height)));
    const camera_center = glm.vec3(0.0, 0.0, 0.0);

    // Calculate the vectors across the horizontal and down the vertical viewport edges
    const viewport_u = glm.vec3(viewport_width, 0.0, 0.0);
    const viewport_v = glm.vec3(0.0, -viewport_height, 0.0);

    const pixel_delta_u = viewport_u.divScalar(@floatFromInt(image_width));
    const pixel_delta_v = viewport_v.divScalar(@floatFromInt(image_height));

    // Calculate the position of the upper left pixel
    const viewport_upper_left = camera_center.sub(glm.vec3(0.0, 0.0, focal_length)).sub(viewport_u.divScalar(2.0)).sub(viewport_v.divScalar(2.0));

    //const pixe00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    const pixel00_loc = viewport_upper_left.add(pixel_delta_u.add(pixel_delta_v).mulScalar(0.5));

    // Render
    const file_name = "image.ppm";

    // Open the file, or create it if it does not exist
    var file = try openOrCreateFile(file_name);
    defer file.close();

    const file_stream = file.writer();
    var bw = std.io.bufferedWriter(file_stream);

    const file_writer = bw.writer();

    try file_writer.print("P3\n{} {}\n255\n", .{ image_width, image_height });

    for (0..image_height) |j| {
        // Print progress, pretty basic and not very elegant.
        const scanlines_remaining = (image_height - j);
        std.debug.print("Scanlines remaining {}\n", .{scanlines_remaining});

        for (0..image_width) |i| {
            const pixel_center = pixel00_loc.add(pixel_delta_u.mulScalar(@floatFromInt(i))).add(pixel_delta_v.mulScalar(@floatFromInt(j)));
            const ray_direction = pixel_center.sub(camera_center);

            const r = ray.Ray.init(camera_center, ray_direction);

            const pixel_color = rayColor(r);

            try color.writeColor(&file_writer, pixel_color);
        }
    }

    try bw.flush();
}

fn openOrCreateFile(file_name: []const u8) !std.fs.File {
    const fs = std.fs.cwd();
    const result = fs.openFile(file_name, .{ .mode = .read_write });
    if (result) |file| {
        return file;
    } else |err| {
        switch (err) {
            // If the file does not exist, create it
            error.FileNotFound => return fs.createFile(file_name, .{}),
            else => return err,
        }
    }
}
