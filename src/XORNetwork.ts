import { AdamW } from "./AdamW.js";
import DEBUG from "./Debug.js";
import { HiddenLayer } from "./HiddenLayer.js";
import { Hyperparameters } from "./Hyperparameters.js";
import { OutputLayer } from "./OutputLayer.js";
import { setSeed } from "./Random.js";
import { crossEntropyLoss, oneHot } from "./Trainer.js";

/** One XOR example: two inputs (each 0 or 1) and the correct answer. */
export interface XORExample {
    readonly inputs: readonly number[];
    readonly answer: number;
}

/**
 * A tiny network that learns the XOR (exclusive or) function, to show
 * backpropagation on a network small enough to follow by hand:
 * 2 inputs -> hidden layer (2 neurons) -> output layer (2 neurons).
 *
 *     input   answer
 *     0, 0    0
 *     0, 1    1
 *     1, 0    1
 *     1, 1    0
 *
 * XOR cannot be learned without a hidden layer: no single straight line
 * separates the 1s from the 0s. The hidden layer re-maps the inputs so that
 * the output layer can separate them.
 *
 * The output layer works like the digit network's: one neuron per class and
 * softmax, so output[0] is the probability that the answer is 0 and
 * output[1] is the probability that it is 1.
 *
 * A trained network typically turns one hidden neuron into "first input but
 * not second" and the other into "second input but not first"; the output
 * layer then answers 1 if either neuron is active, which is XOR.
 * printResults() shows the hidden activations so this can be seen.
 *
 * The hidden neurons use ReLU. With only two hidden neurons and inputs of 0
 * and 1, a ReLU neuron often starts with both weights negative, outputs 0
 * for every input and never learns (a "dead" ReLU). Many random starts
 * therefore get stuck: if training does not reach 4/4, try a different
 * randomSeed. Settings that work: learningRate 0.01, epochs 2000,
 * randomSeed 9. Passing "sigmoid" to the hidden layer instead avoids dead
 * neurons, and most random starts then succeed.
 */
export class XORNetwork {
    /** Number of inputs. */
    static readonly INPUT_SIZE = 2;

    /** Number of hidden neurons. */
    static readonly HIDDEN_SIZE = 2;

    /** Number of outputs: one for the answer 0 and one for the answer 1. */
    static readonly NUM_CLASSES = 2;

    /** The complete XOR truth table, which is also the whole training set. */
    static readonly EXAMPLES: readonly XORExample[] = [
        { inputs: [0, 0], answer: 0 },
        { inputs: [0, 1], answer: 1 },
        { inputs: [1, 0], answer: 1 },
        { inputs: [1, 1], answer: 0 },
    ];

    private readonly _hiddenLayer: HiddenLayer;
    private readonly _outputLayer: OutputLayer;

    /**
     * @param hyperparameters Settings for the network and training. Uses
     * randomSeed, epochs and the AdamW settings; batchSize and dataSplit are
     * ignored, because every epoch is one batch of all four examples.
     */
    constructor(readonly hyperparameters: Hyperparameters) {
        this._hiddenLayer = new HiddenLayer(XORNetwork.INPUT_SIZE, XORNetwork.HIDDEN_SIZE, "relu");
        this._outputLayer = new OutputLayer(this._hiddenLayer.numNeurons, XORNetwork.NUM_CLASSES);

        //Build the network
        this.build();
    }

    /**
     * Builds both layers, creating their weights and biases with random
     * starting values. Must be called before update().
     *
     * The random number generator is first reset to the hyperparameters'
     * randomSeed, so the same seed always gives the same starting weights.
     *
     * When DEBUG.XOR_WEIGHTS is set, the starting weights and biases are logged.
     */
    build(): void {
        setSeed(this.hyperparameters.randomSeed);
        this._hiddenLayer.build();
        this._outputLayer.build();

        if (DEBUG.XOR_WEIGHTS) {
            this.printWeights();
        }
    }

    /**
     * Logs the weights and biases of both layers as tables, with one row per
     * neuron and one column per incoming connection, plus the neuron's bias.
     * The value in row n, column i is the weight neuron n gives to input i.
     *
     * @throws Error if build() has not been called yet.
     */
    printWeights(): void {
        const hiddenTable: Record<string, Record<string, number>> = {};
        this._hiddenLayer.weights.forEach((neuronWeights, n) => {
            const row: Record<string, number> = {};
            neuronWeights.forEach((weight, i) => {
                row[`from input ${i}`] = round(weight);
            });
            row["bias"] = round(this._hiddenLayer.biases[n]!);
            hiddenTable[`hidden ${n}`] = row;
        });

        const outputTable: Record<string, Record<string, number>> = {};
        this._outputLayer.weights.forEach((neuronWeights, n) => {
            const row: Record<string, number> = {};
            neuronWeights.forEach((weight, i) => {
                row[`from hidden ${i}`] = round(weight);
            });
            row["bias"] = round(this._outputLayer.biases[n]!);
            outputTable[`output ${n} (answer ${n})`] = row;
        });

        console.log("Hidden layer weights and biases:");
        console.table(hiddenTable);
        console.log("Output layer weights and biases:");
        console.table(outputTable);
    }

    /**
     * Feeds two inputs forward through the network.
     *
     * When DEBUG.XOR_UPDATE is set, logs the inputs, each hidden neuron's
     * activation, the output logits and the output probabilities.
     *
     * @param inputs The two inputs, e.g. [0, 1].
     * @returns Two probabilities: [probability the answer is 0, probability it is 1].
     * @throws RangeError if there are not two inputs.
     * @throws Error if build() has not been called yet.
     */
    update(inputs: readonly number[]): number[] {
        const hiddenOutput = this._hiddenLayer.update(inputs);
        const probabilities = this._outputLayer.update(hiddenOutput);

        if (DEBUG.XOR_UPDATE) {
            console.log(`Input: ${formatVector(inputs)}`);
            console.log(`  Hidden activations: ${formatVector(hiddenOutput)}`);
            console.log(`  Output logits:      ${formatVector(this._outputLayer.logits)}`);
            console.log(`  Output P(0), P(1):  ${formatVector(probabilities)}`);
        }
        return probabilities;
    }

    /**
     * Works out every neuron's error for the most recent inputs passed to
     * update(), starting at the output and working backwards:
     *
     *     output layer -> errors from comparing its output with the target
     *     hidden layer -> errors passed back through the output layer's weights,
     *                     multiplied by the ReLU slope (1 if the neuron
     *                     was active, 0 if it output 0)
     *
     * When DEBUG.XOR_ERROR_SIGNAL is set, logs the target and the error
     * signals of every neuron.
     *
     * @param target The desired output as a one-hot vector: [1, 0] for the
     * answer 0, [0, 1] for the answer 1.
     * @throws RangeError if target is not a one-hot vector of two values.
     * @throws Error if update() has not been called yet.
     */
    backpropagate(target: readonly number[]): void {
        const outputErrors = this._outputLayer.calculateErrors(target);
        const errorsForHidden = this._outputLayer.calculatePreviousLayerErrors(outputErrors);
        const hiddenErrors = this._hiddenLayer.calculateErrors(errorsForHidden);
        // The hidden layer's inputs are the XOR inputs, which are not learned,
        // so there is nothing further back to pass errors to.

        if (DEBUG.XOR_ERROR_SIGNAL) {
            console.log(`Target: ${formatVector(target)}`);
            console.log(`  Output error signals (probability - target): ${formatVector(outputErrors)}`);
            console.log(`  Errors passed back to hidden layer:          ${formatVector(errorsForHidden)}`);
            console.log(`  Hidden error signals (x activation slope):   ${formatVector(hiddenErrors)}`);
        }
    }

    /**
     * Asks both layers to calculate the gradients of their weights and biases
     * from their stored errors and inputs. Call backpropagate() first.
     *
     * @param batchAverage true to average the new gradients with the stored
     * ones (a running mean over the batch); false to replace them.
     * @throws Error if update() or backpropagate() has not been called yet.
     */
    calculateGradients(batchAverage: boolean): void {
        this._outputLayer.calculateGradients(this._outputLayer.errors, batchAverage);
        this._hiddenLayer.calculateGradients(this._hiddenLayer.errors, batchAverage);
    }

    /**
     * Takes one optimization step, updating both layers' weights and biases
     * from their stored (batch-averaged) gradients.
     *
     * @param optimizer The AdamW optimizer. Use the same one for the whole of training.
     * @throws Error if calculateGradients() has not been called yet.
     */
    updateWeights(optimizer: AdamW): void {
        optimizer.step();
        this._outputLayer.updateWeights(optimizer);
        this._outputLayer.updateBiases(optimizer);
        this._hiddenLayer.updateWeights(optimizer);
        this._hiddenLayer.updateBiases(optimizer);
    }

    /**
     * Builds the network and trains it on the four XOR examples for the
     * number of epochs in the hyperparameters. Each epoch is a single batch:
     *
     *   For each of the four examples:
     *     1. Feed the inputs forward.
     *     2. Measure the cross-entropy loss against the answer.
     *     3. Backpropagate, so both layers have their errors for this example.
     *     4. Calculate the gradients, averaging them over the four examples.
     *   Then:
     *     5. The AdamW optimizer updates the weights and biases once, using
     *        the averaged gradients.
     *
     * The average loss is logged every logInterval epochs, and after the last
     * epoch. A network guessing 50/50 scores -log(0.5), about 0.69; as it
     * learns XOR the loss falls towards 0.
     *
     * @param logInterval How many epochs between loss reports.
     */
    train(logInterval = 100): void {
        const optimizer = new AdamW(this.hyperparameters);
        const epochs = this.hyperparameters.epochs;

        for (let epoch = 1; epoch <= epochs; epoch++) {
            let lossTotal = 0;
            XORNetwork.EXAMPLES.forEach((example, index) => {
                // Step 1: forward pass.
                const probabilities = this.update(example.inputs);

                // Step 2: loss.
                lossTotal += crossEntropyLoss(probabilities, example.answer);

                // Step 3: backpropagate the errors from the desired output.
                this.backpropagate(oneHot(example.answer, XORNetwork.NUM_CLASSES));

                // Step 4: gradients. The first example replaces the previous
                // epoch's gradients; the rest are averaged in.
                this.calculateGradients(index > 0);
            });

            // Step 5: one weight update per epoch.
            this.updateWeights(optimizer);

            if (epoch % logInterval === 0 || epoch === epochs) {
                const loss = lossTotal / XORNetwork.EXAMPLES.length;
                console.log(`Epoch ${epoch}/${epochs}: average loss ${loss.toFixed(4)}`);
            }
        }
    }

    /**
     * Logs the network's output for each of the four examples, how many it
     * gets right, and the learned weights and biases.
     *
     * The prediction is the answer with the higher probability.
     *
     * @throws Error if build() (or train()) has not been called yet.
     */
    printResults(): void {
        const table: Record<string, Record<string, string | number>> = {};
        let correct = 0;
        for (const example of XORNetwork.EXAMPLES) {
            const probabilities = this.update(example.inputs);
            const predicted = probabilities[1]! > probabilities[0]! ? 1 : 0;
            if (predicted === example.answer) {
                correct++;
            }
            table[`input ${example.inputs.join(", ")}`] = {
                "hidden": this._hiddenLayer.activations.map(a => a.toFixed(3)).join(", "),
                "P(0)": probabilities[0]!.toFixed(3),
                "P(1)": probabilities[1]!.toFixed(3),
                "predicted": predicted,
                "answer": example.answer,
            };
        }
        console.table(table);
        console.log(`Correct: ${correct}/${XORNetwork.EXAMPLES.length}`);

        this.printWeights();
    }
}

/**
 * Rounds a number to three decimal places. Returning a number rather than a
 * string keeps console.table from wrapping each value in quotes.
 */
function round(value: number): number {
    return Math.round(value * 1000) / 1000;
}

/** Formats an array of numbers to three decimal places, e.g. "[0.120, -1.500]". */
function formatVector(values: readonly number[]): string {
    return `[${values.map(value => value.toFixed(3)).join(", ")}]`;
}
