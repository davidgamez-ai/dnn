import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { PNG } from "pngjs";
import DEBUG from "./Debug.js";

/** The first 8 bytes of every PNG file. */
const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

/**
 * Loads image files as grayscale pixel values that a neural network can use.
 * Only PNG images are supported.
 */
export class ImageLoader {
    /**
     * Reads an image and returns its pixels as a 2D array indexed [row][column],
     * with each value normalized to the range 0-1 (0 = black, 1 = white).
     * Colour images are converted to grayscale.
     *
     * @param filePath Path to the image, absolute or relative to the current working directory.
     * @throws Error if the file does not exist or is not a PNG image.
     */
    load(filePath: string): number[][] {
        const fullPath = resolve(filePath);

        let isFile: boolean;
        try {
            isFile = statSync(fullPath).isFile();
        } catch {
            throw new Error(`Image file not found: ${fullPath}`);
        }
        if (!isFile) {
            throw new Error(`Not a file: ${fullPath}`);
        }

        // Check the file's contents rather than trusting its extension.
        const buffer = readFileSync(fullPath);
        if (!buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
            throw new Error(`Not a PNG image: ${fullPath}`);
        }

        // pngjs decodes every PNG to 8-bit RGBA: 4 bytes per pixel, row by row.
        const { width, height, data } = PNG.sync.read(buffer);

        const pixels: number[][] = [];
        for (let y = 0; y < height; y++) {
            const row: number[] = [];
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                // Standard luminance weights; for gray pixels (R = G = B) this is just the gray value.
                const gray = 0.299 * data[i]! + 0.587 * data[i + 1]! + 0.114 * data[i + 2]!;
                row.push(gray / 255);
            }
            pixels.push(row);
        }

        if (DEBUG.IMAGE_LOAD) {
            console.table(pixels);
        }
        return pixels;
    }

    /**
     * Converts a 2D array of pixels into a 1D array, which is the shape the
     * network's input layer expects. Rows are placed one after another, so
     * an 8x8 image becomes 64 values: row 0, then row 1, and so on.
     *
     * @param pixels Pixel values indexed [row][column], as returned by load().
     * @throws RangeError if the rows are not all the same length.
     */
    flatten(pixels: readonly (readonly number[])[]): number[] {
        const width = pixels[0]?.length ?? 0;
        const flat: number[] = [];
        for (const row of pixels) {
            if (row.length !== width) {
                throw new RangeError(`Every row must have ${width} pixels, but found a row with ${row.length}`);
            }
            for (const value of row) {
                flat.push(value);
            }
        }
        return flat;
    }
}
