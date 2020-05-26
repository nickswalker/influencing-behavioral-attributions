import "phaser"
import {Gridworld, GridworldState, TerrainMap, TerrainType} from "./gridworld-mdp"

const sceneConfig: Phaser.Types.Scenes.SettingsConfig = {
    active: false,
    visible: false,
    key: 'Game',
};

export class GridworldGame {
    gameWidth: number
    gameHeight: number
    container: HTMLElement
    mdp: Gridworld
    scene: any
    state: GridworldState
    assetsPath: string
    game: Phaser.Game
    _stateToDraw: GridworldState
    _scoreToDraw: number
    _timeToDraw: number
    constructor (
        start_grid: TerrainMap,
        container: HTMLElement,
        tileSize = 128,
        gameWidth = tileSize*start_grid[0].length,
        gameHeight = tileSize*start_grid.length,
        ANIMATION_DURATION = 500,
        TIMESTEP_DURATION = 600,
        assetsLoc = "./assets/",

    ) {
        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;
        this.container = container;

        this.mdp = new Gridworld(start_grid);
        this.state = this.mdp.getStartState();

        this.assetsPath = assetsLoc;
        this.scene = new GridworldScene(this, tileSize)
    }

    init () {
        let gameConfig: Phaser.Types.Core.GameConfig = {
            type: Phaser.CANVAS,
            width: this.gameWidth,
            height: this.gameHeight,
            scene: this.scene,
            parent: this.container,
            audio: {
                noAudio: true
            }
        };
        this.game = new Phaser.Game(gameConfig);
    }

    drawState(state: GridworldState) {
        this._stateToDraw = state;
    }

    setAction(player_index: number, action: any) {

    }

    drawScore(score: number) {
        this._scoreToDraw = score;
    }

    drawTimeLeft(time_left: number) {
        this._timeToDraw = time_left;
    }

    close (msg: string) {
        this.game.renderer.destroy();
        this.game.loop.stop();
        // this.game.canvas.remove();
        this.game.destroy(false);
    }

}

type SpriteMap = {[key: string]: any}

export class GridworldScene extends Phaser.Scene {
    gameParent: GridworldGame
    tileSize: number
    sceneSprite: object
    mdp: Gridworld
    constructor(gameParent: GridworldGame, tileSize: number) {
        super(sceneConfig);
        this.gameParent = gameParent
        this.tileSize = tileSize
        this.mdp = this.gameParent.mdp;
    }

    preload() {
        this.load.atlas("tiles",
            this.gameParent.assetsPath+"tiles.png",
            this.gameParent.assetsPath+"tiles.json");
    }
    create(data: object) {
        this.sceneSprite = {};
        this.drawLevel();
        //this._drawState(this.gameParent.state, this.sprites);
    }
    drawLevel() {
        //draw tiles
        const terrain_to_img: {[key in TerrainType]: string} = {
            [TerrainType.Empty]: 'white.png',
            [TerrainType.Goal]: 'green.png',
            [TerrainType.Fire]: 'red.png',
            [TerrainType.Wall]: "black.png"
        };
        let pos_dict = this.mdp.terrain
        for (let y = 0; y < pos_dict.length; y++) {
            for (let x = 0; x < pos_dict[0].length; x++) {
                const type = pos_dict[y][x]
                const tile = this.add.sprite(
                    this.tileSize * x,
                    this.tileSize * y,
                    "tiles",
                    terrain_to_img[type]
                );
                tile.setDisplaySize(this.tileSize, this.tileSize);
                tile.setOrigin(0);
            }
        }
    }
    _drawState(state: GridworldState, sprites: SpriteMap) {
        sprites = typeof(sprites) === 'undefined' ? {} : sprites;

    }

    _drawScore(score: number, sprites: SpriteMap) {
        const scoreString = "Score: "+ score.toString();
        if (typeof(sprites['score']) !== 'undefined') {
            sprites['score'].setText(score);
        }
        else {
            sprites['score'] = this.add.text(
                5, 25, scoreString,
                {
                    font: "20px Arial",
                    fill: "yellow",
                    align: "left"
                }
            )
        }
    }
    _drawTimeLeft(timeLeft: number, sprites: SpriteMap) {
        const time_left = "Time Left: "+ timeLeft.toString();
        if (typeof(sprites['time_left']) !== 'undefined') {
            sprites['time_left'].setText(time_left);
        }
        else {
            sprites['time_left'] = this.add.text(
                5, 5, time_left,
                {
                    font: "20px Arial",
                    fill: "yellow",
                    align: "left"
                }
            )
        }
    }
    update(time: number, delta: number) {
        return
        if (typeof(this.gameParent.state) !== 'undefined') {
            let state = this.gameParent.state;
            delete this.gameParent._stateToDraw;
            // redraw = true;
            this._drawState(state, this.sceneSprite);
        }
        if (typeof(this.gameParent._scoreToDraw) !== 'undefined') {
            let score = this.gameParent._scoreToDraw;
            delete this.gameParent._scoreToDraw;
            this._drawScore(score, this.sceneSprite);
        }
        if (typeof(this.gameParent._timeToDraw) !== 'undefined') {
            let timeLeft = this.gameParent._timeToDraw;
            delete this.gameParent._timeToDraw;
            this._drawTimeLeft(timeLeft, this.sceneSprite);
        }

    }
}