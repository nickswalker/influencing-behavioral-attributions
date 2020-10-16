const path = require('path');
// https://2ality.com/2020/04/webpack-typescript.html
module.exports = {
    mode: "development",
    entry: {
        "interactive": "./src/interactive-index.ts",
        "browse-batch": "./src/browse-batch-index.ts",
        "sorting": './src/sorting-index.ts',
        "show-groupings": './src/show-groupings-index.ts',
    },
    devtool: 'inline-source-map',
    resolve: {
        extensions: [ '.tsx', '.ts', '.js' ],
    },
    module: {
        rules: [
            // all files with a `.ts` or `.tsx` extension will be handled by `ts-loader`
            { test: /\.tsx?$/, loader: "ts-loader",  options: {
                    transpileOnly: true,
                    experimentalWatchApi: true,
                }, },
        ],
    },
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    optimization: {
        splitChunks: {
            cacheGroups: {
                // In dev mode, we want all vendor (node_modules) to go into a chunk,
                // so building main.js is faster.
                vendors: {
                    test: /[\\/]node_modules[\\/]/,
                    name: "vendors",
                    // Exclude pre-main dependencies going into vendors, as doing so
                    // will result in webpack only loading pre-main once vendors loaded.
                    // But pre-main is the one loading vendors.
                    // Currently undocument feature:  https://github.com/webpack/webpack/pull/6791
                    chunks: chunk => chunk.name !== "pre-main.min"
                }
            }
        }
    },
    externals: {
        "phaser": "Phaser"
    }
};