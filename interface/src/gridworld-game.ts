import "phaser"
import {Direction, Gridworld, GridworldState, TerrainMap, TerrainType} from "./gridworld-mdp"

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

    setAction(player_index: number, action: Direction) {
        const nextState = this.mdp.transition(this.state, action);
        this._stateToDraw = nextState;
        this.state = nextState
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
    interactive: boolean
    inputDelayTimeout: Phaser.Time.TimerEvent
    ANIMATION_DURATION = 500
    TIMESTEP_DURATION = 600
    constructor(gameParent: GridworldGame, tileSize: number, interactive: boolean = true) {
        super(sceneConfig);
        this.gameParent = gameParent
        this.tileSize = tileSize
        this.mdp = this.gameParent.mdp
        this.interactive = interactive
        this.inputDelayTimeout = null
    }

    preload() {
        this.load.atlas("tiles",
            this.gameParent.assetsPath+"tiles.png",
            this.gameParent.assetsPath+"tiles.json");
        this.load.image("agent", this.gameParent.assetsPath+"agent.png")
    }
    create(data: object) {
        this.sceneSprite = {};
        this.drawLevel();
        this._drawState(this.gameParent.state, this.sceneSprite);
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
        sprites["agents"] = typeof(sprites["agents"]) === 'undefined' ? {} : sprites["agents"];
        for (let p = 0; p < state.agentPositions.length; p++){
            const agentPosition = state.agentPositions[p]
            // Flip Y to make lower-left the origin
            let [drawX, drawY] = [agentPosition.x, this.mdp.height - agentPosition.y - 1]
            if (typeof(sprites['agents'][p]) === 'undefined') {
                const agent = this.add.sprite(this.tileSize * drawX, this.tileSize * drawY, "agent");
                agent.setDisplaySize(this.tileSize, this.tileSize)
                agent.setOrigin(0)
                sprites['agents'][p] = agent;
            } else {
                const agent = sprites['agents'][p]
                this.tweens.add({
                    targets: [agent],
                    x: this.tileSize*drawX,
                    y: this.tileSize*drawY,
                    duration: this.ANIMATION_DURATION,
                    ease: 'Linear',
                    onComplete: (tween, target, player) => {
                        target[0].setPosition(this.tileSize*drawX, this.tileSize*drawY);
                    }
                })
            }

        }
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
        if (this.inputDelayTimeout && this.inputDelayTimeout.getOverallProgress() == 1.0) {
            this.inputDelayTimeout = null
        }
        const acceptingInput = this.inputDelayTimeout == null
        if (this.interactive && acceptingInput) {
            const cursors = this.input.keyboard.createCursorKeys();
            let action: Direction = null
            if (cursors.left.isDown) {
                action = Direction.WEST
            } else if (cursors.right.isDown) {
                action = Direction.EAST
            } else if (cursors.up.isDown) {
                action = Direction.NORTH
            } else if (cursors.down.isDown) {
                action = Direction.SOUTH
            }
            if (action !== null) {
                this.gameParent.setAction(0, action)
                this.inputDelayTimeout = this.time.addEvent({delay: 500})
            }
        }
        if (typeof(this.gameParent._stateToDraw) !== 'undefined') {
            let state = this.gameParent._stateToDraw;
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