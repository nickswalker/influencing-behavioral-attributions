const path = require('path');
// https://2ality.com/2020/04/webpack-typescript.html
module.exports = {
    mode: "development",
    entry: './src/index.ts',
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
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist'),
    },
    optimization: {
        splitChunks: {
            // include all types of chunks
            chunks: 'all'
        }
    },
    externals: {
        "phaser": "Phaser"
    }
};