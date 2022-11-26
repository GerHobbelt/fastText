const { FastText } = require('./fasttext_node');

describe("FastText WebAssemply", () => {
	let fastText;

	beforeEach(async () => {
		fastText = await FastText.from("model-2018-11-12.bin");
	});

	const table = [
		["Exception when compiling", "bug"],
		["Idea to improve the product", "enhancement"],
		["I have a question", "question"],
	];
	test.each(table)(".predict(%s)", (text, expectedLabel) => {
		const [[label, prob]] = fastText.predict(text);
		expect(label).toBe(expectedLabel);
		expect(prob).toBeGreaterThan(0.5);
	})
});
