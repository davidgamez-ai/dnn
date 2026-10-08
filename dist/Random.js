/**
 * Returns a random number from the standard normal distribution
 * (mean 0, standard deviation 1), using the Box-Muller transform.
 */
export function randomNormal() {
    // 1 - Math.random() is in (0, 1], which avoids taking the log of 0.
    const u1 = 1 - Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
//# sourceMappingURL=Random.js.map