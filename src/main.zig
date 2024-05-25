const std = @import("std");

const color = @import("color.zig");
const glm = @import("glm.zig");

pub fn main() !void {
    // Image
    const image_width = 256;
    const image_height = 256;

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
            const r: f32 = (@as(f32, @floatFromInt(i)) / (image_width - 1));
            const g: f32 = (@as(f32, @floatFromInt(j)) / (image_height - 1));
            const b: f32 = 0.0;

            const pixel_color = glm.vec3(r, g, b);
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
