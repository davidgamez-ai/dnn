import DEBUG from "./Debug.js";
import { HiddenLayer } from "./HiddenLayer.js";
import { OutputLayer } from "./OutputLayer.js";
import { setSeed } from "./Random.js";
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
    hyperparameters;
    /** Number of inputs. */
    static INPUT_SIZE = 2;
    /** Number of hidden neurons. */
    static HIDDEN_SIZE = 2;
    /** Number of outputs: one for the answer 0 and one for the answer 1. */
    static NUM_CLASSES = 2;
    /** The complete XOR truth table, which is also the whole training set. */
    static EXAMPLES = [
        { inputs: [0, 0], answer: 0 },
        { inputs: [0, 1], answer: 1 },
        { inputs: [1, 0], answer: 1 },
        { inputs: [1, 1], answer: 0 },
    ];
    _hiddenLayer;
    _outputLayer;
    /**
     * @param hyperparameters Settings for the network. Uses randomSeed, which
     * sets the starting weights. Training is done by XORTrainer.
     */
    constructor(hyperparameters) {
        this.hyperparameters = hyperparameters;
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
    build() {
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
    printWeights() {
        const hiddenTable = {};
        this._hiddenLayer.weights.forEach((neuronWeights, n) => {
            const row = {};
            neuronWeights.forEach((weight, i) => {
                row[`from input ${i}`] = round(weight);
            });
            row["bias"] = round(this._hiddenLayer.biases[n]);
            hiddenTable[`hidden ${n}`] = row;
        });
        const outputTable = {};
        this._outputLayer.weights.forEach((neuronWeights, n) => {
            const row = {};
            neuronWeights.forEach((weight, i) => {
                row[`from hidden ${i}`] = round(weight);
            });
            row["bias"] = round(this._outputLayer.biases[n]);
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
    update(inputs) {
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
    backpropagate(target) {
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
     *
     * When DEBUG.XOR_GRADIENTS is set, logs the weight and bias gradients of
     * both layers. Each weight gradient is error[neuron] × input[input], and
     * each bias gradient is error[neuron].
     *
     * @throws Error if update() or backpropagate() has not been called yet.
     */
    calculateGradients(batchAverage) {
        this._outputLayer.calculateGradients(this._outputLayer.errors, batchAverage);
        this._hiddenLayer.calculateGradients(this._hiddenLayer.errors, batchAverage);
        if (DEBUG.XOR_GRADIENTS) {
            const count = this._outputLayer.gradientCount;
            console.log(`Gradients (averaged over ${count} example${count === 1 ? "" : "s"}):`);
            console.log(`  Output weight gradients [neuron][input]: ${formatMatrix(this._outputLayer.weightGradients)}`);
            console.log(`  Output bias gradients:                   ${formatVector(this._outputLayer.biasGradients)}`);
            console.log(`  Hidden weight gradients [neuron][input]: ${formatMatrix(this._hiddenLayer.weightGradients)}`);
            console.log(`  Hidden bias gradients:                   ${formatVector(this._hiddenLayer.biasGradients)}`);
        }
    }
    /**
     * Takes one optimization step, updating both layers' weights and biases
     * from their stored (batch-averaged) gradients.
     *
     * @param optimizer The AdamW optimizer. Use the same one for the whole of training.
     * @throws Error if calculateGradients() has not been called yet.
     */
    updateWeights(optimizer) {
        optimizer.step();
        this._outputLayer.updateWeights(optimizer);
        this._outputLayer.updateBiases(optimizer);
        this._hiddenLayer.updateWeights(optimizer);
        this._hiddenLayer.updateBiases(optimizer);
    }
    /**
     * Logs the network's output for each of the four examples, how many it
     * gets right, and the learned weights and biases.
     *
     * The prediction is the answer with the higher probability.
     *
     * @throws Error if build() has not been called yet.
     */
    printResults() {
        const table = {};
        let correct = 0;
        for (const example of XORNetwork.EXAMPLES) {
            const probabilities = this.update(example.inputs);
            const predicted = probabilities[1] > probabilities[0] ? 1 : 0;
            if (predicted === example.answer) {
                correct++;
            }
            table[`input ${example.inputs.join(", ")}`] = {
                "hidden": this._hiddenLayer.activations.map(a => a.toFixed(3)).join(", "),
                "P(0)": probabilities[0].toFixed(3),
                "P(1)": probabilities[1].toFixed(3),
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
function round(value) {
    return Math.round(value * 1000) / 1000;
}
/** Formats an array of numbers to three decimal places, e.g. "[0.120, -1.500]". */
function formatVector(values) {
    return `[${values.map(value => value.toFixed(3)).join(", ")}]`;
}
/** Formats a matrix row by row, e.g. "[[0.120, -1.500], [2.000, 0.300]]". */
function formatMatrix(rows) {
    return `[${rows.map(formatVector).join(", ")}]`;
}
//# sourceMappingURL=XORNetwork.js.map