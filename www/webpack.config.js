const CopyPlugin = require("copy-webpack-plugin");
const path = require("path");

module.exports = {
  entry: "./bootstrap.js",
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "production",
  experiments: {
    syncWebAssembly: true, // Enable WebAssembly experiments
  },
  watchOptions: {
    aggregateTimeout: 600,
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: "index.html", to: "index.html" },
      ],
    }),
  ],
  cache: false
};
