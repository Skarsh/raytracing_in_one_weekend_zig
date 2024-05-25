const std = @import("std");
const glm = @import("glm.zig");

pub const Color = glm.Vec3;

pub fn writeColor(writer: anytype, pixel_color: Color) !void {
    const r = pixel_color.vals[0];
    const g = pixel_color.vals[1];
    const b = pixel_color.vals[2];

    // Translate the [0, 1] component values to the byte range [0, 255]
    const r_byte: u8 = @intFromFloat(255.999 * r);
    const g_byte: u8 = @intFromFloat(255.999 * g);
    const b_byte: u8 = @intFromFloat(255.999 * b);

    try writer.print("{} {} {}\n", .{ r_byte, g_byte, b_byte });
}
