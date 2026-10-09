import { readdirSync } from "node:fs";
import { join } from "node:path";
import DEBUG from "./Debug.js";
import { Hyperparameters } from "./Hyperparameters.js";
import { ImageLoader } from "./ImageLoader.js";
import { Network } from "./Network.js";
import { random, setSeed } from "./Random.js";

/** An image file paired with the digit it shows. */
export interface LabelledImage {
    /** Path to the image file, e.g. "data/images/3/0003.png" (backslashes on Windows). */
    readonly fileName: string;
    /** The digit (0-9) the image represents. */
    readonly digit: number;
}

/**
 * Trains a network on the digit images.
 */
export class Trainer {
    private readonly _hyperparameters: Hyperparameters;
    private readonly _network: Network;
    private readonly _imageLoader = new ImageLoader();
    private _train: LabelledImage[] = [];
    private _validation: LabelledImage[] = [];
    private _test: LabelledImage[] = [];

    /**
     * Creates the network to be trained and builds it, so its weights and
     * biases are ready to use.
     *
     * @param hyperparameters Settings for training, including how to split the data and the random seed.
     */
    constructor(readonly hyperparameters: Hyperparameters) {
        this._hyperparameters = hyperparameters;
        this._network = new Network(hyperparameters);
        this._network.build();
    }

    /**
     * Finds every image in the images folder and divides them into training,
     * validation and test sets, using the fractions in the hyperparameters'
     * dataSplit. Each image is labelled with its digit, which is the name of
     * the folder it is in. Calling it again replaces the previous sets.
     *
     * The images are shuffled before splitting. On disk they are grouped by
     * digit, so without shuffling the test set would contain only 8s and 9s.
     * The random number generator is first reset to the hyperparameters'
     * randomSeed, so the same seed always gives the same split.
     *
     * @param imagesDirectory Folder containing one subfolder per digit (0-9), each holding PNG images.
     */
    loadData(imagesDirectory = "data/images"): void {
        const images: LabelledImage[] = [];
        for (const folder of readdirSync(imagesDirectory, { withFileTypes: true })) {
            if (!folder.isDirectory() || !/^[0-9]$/.test(folder.name)) {
                continue;
            }
            const digit = Number(folder.name);
            const folderPath = join(imagesDirectory, folder.name);
            for (const file of readdirSync(folderPath)) {
                if (file.toLowerCase().endsWith(".png")) {
                    images.push({ fileName: join(folderPath, file), digit });
                }
            }
        }

        // Sort first so the starting order does not depend on how the file system lists files.
        images.sort((a, b) => a.fileName.localeCompare(b.fileName));
        setSeed(this._hyperparameters.randomSeed);
        shuffle(images);

        const { train, validation } = this._hyperparameters.dataSplit;
        const trainCount = Math.round(images.length * train);
        const validationCount = Math.round(images.length * validation);
        // The test set takes whatever is left, so rounding never loses or duplicates an image.
        this._train = images.slice(0, trainCount);
        this._validation = images.slice(trainCount, trainCount + validationCount);
        this._test = images.slice(trainCount + validationCount);

        console.log(`Training images: ${this._train.length}`);
        console.log(`Validation images: ${this._validation.length}`);
        console.log(`Test images: ${this._test.length}`);
    }

    /**
     * Works through the training images once. Each image is loaded, flattened
     * and fed into the network, and the network's output is compared with the
     * image's digit using cross-entropy loss. The images are grouped into
     * batches of batchSize (from the hyperparameters), and the average loss of
     * each batch is logged. The last batch may be smaller if the images do not
     * divide evenly.
     *
     * The weights are not updated yet, so the loss stays at roughly the same level.
     *
     * @throws Error if loadData() has not been called, or found no training images.
     */
    train(): void {
        if (this._train.length === 0) {
            throw new Error("No training images; call loadData() first");
        }
        const batchSize = this._hyperparameters.batchSize;
        const numBatches = Math.ceil(this._train.length / batchSize);

        let batchCounter = 0;
        let batchLossTotal = 0;
        let batchNumber = 0;
        for (const image of this._train) {
            const pixels = this._imageLoader.flatten(this._imageLoader.load(image.fileName));
            const probabilities = this._network.calculate(pixels);
            batchLossTotal += crossEntropyLoss(probabilities, image.digit);
            batchCounter++;

            const errorSignal = outputErrorSignal(probabilities, image.digit);
            if (DEBUG.OUTPUT_ERROR_SIGNAL) {
                console.log(`Error signal (digit ${image.digit}): [${errorSignal.map(e => e.toFixed(4)).join(", ")}]`);
            }

            // End of a batch: either it is full, or this is the last image.
            if (batchCounter === batchSize || image === this._train[this._train.length - 1]) {
                batchNumber++;
                const batchLoss = batchLossTotal / batchCounter;
                console.log(`Batch ${batchNumber}/${numBatches}: loss ${batchLoss.toFixed(4)}`);
                batchCounter = 0;
                batchLossTotal = 0;
            }
        }
    }

    /** The network being trained. */
    get network(): Network {
        return this._network;
    }

    /** Images the network learns from. */
    get trainingSet(): readonly LabelledImage[] {
        return this._train;
    }

    /** Images used to check progress during training. */
    get validationSet(): readonly LabelledImage[] {
        return this._validation;
    }

    /** Images held back to measure final performance. */
    get testSet(): readonly LabelledImage[] {
        return this._test;
    }
}

/**
 * Cross-entropy loss for one example: -log of the probability the network
 * gave to the correct digit. It is 0 when the network is certain and right,
 * and grows without limit as the correct digit's probability approaches 0.
 * A network guessing evenly across 10 digits scores -log(0.1), about 2.30.
 *
 * @param probabilities The network's output, one probability per digit.
 * @param digit The correct digit (the label).
 */
function crossEntropyLoss(probabilities: readonly number[], digit: number): number {
    const probability = probabilities[digit];
    if (probability === undefined) {
        throw new RangeError(`Label ${digit} has no matching output; the network has ${probabilities.length} outputs`);
    }
    // A probability of exactly 0 would give -log(0) = Infinity, so use a tiny minimum.
    return -Math.log(Math.max(probability, 1e-12));
}

/**
 * The error signal (delta) for each output neuron: how much the loss would
 * change if that neuron's logit changed. With softmax and cross-entropy loss
 * together this simplifies to probability - target, where the target is 1 for
 * the correct digit and 0 for every other digit (a "one-hot" encoding).
 *
 * The correct digit's signal is negative (its probability should go up) and
 * every other signal is positive (their probabilities should go down).
 * Backpropagation starts from these values.
 *
 * @param probabilities The network's output, one probability per digit.
 * @param digit The correct digit (the label).
 */
function outputErrorSignal(probabilities: readonly number[], digit: number): number[] {
    return probabilities.map((probability, d) => probability - (d === digit ? 1 : 0));
}

/**
 * Shuffles an array in place into a random order, using the Fisher-Yates
 * algorithm, which makes every order equally likely.
 */
function shuffle<T>(items: T[]): void {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [items[i], items[j]] = [items[j]!, items[i]!];
    }
}
