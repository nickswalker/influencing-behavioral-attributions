import "phaser"
import {Direction, Gridworld, GridworldState, Position, TerrainMap, TerrainType} from "./gridworld-mdp"

const terrain_to_img: { [key in TerrainType]: string[] } = {
    [TerrainType.Empty]: ['colors','sol_base2.png'],
    [TerrainType.Goal]: ['landmarks','door.png'],
    [TerrainType.Fire]: ['landmarks','fire.png'],
    [TerrainType.Wall]: ['colors', "sol_base02_full.png"]
};

export class GridworldGame {
    private gameWidth: number
    private gameHeight: number
    container: HTMLElement
    mdp: Gridworld
    scene: any
    state: GridworldState
    assetsPath: string
    game: Phaser.Game
    _stateToDraw: GridworldState
    _animated: boolean
    interactive: boolean
    displayTrajectory: number[]
    sceneCreatedDelegate?: () => void

    constructor(
        start_grid: TerrainMap,
        container: HTMLElement,
        tileSize = 128,
        gameWidth = tileSize * start_grid[0].length,
        gameHeight = tileSize * start_grid.length,
        assetsLoc = "./assets/",
        interactive: boolean = true
    ) {
        this.gameWidth = gameWidth;
        this.gameHeight = gameHeight;
        this.container = container;

        this.mdp = new Gridworld(start_grid);
        this.state = this.mdp.getStartState();

        this.assetsPath = assetsLoc;
        this.scene = new GridworldScene(this, this.mdp.terrain, tileSize)
        this.interactive = interactive
    }

    init() {
        let gameConfig: Phaser.Types.Core.GameConfig = {
            type: Phaser.CANVAS,
            backgroundColor: "#93a1a1",
            width: this.gameWidth,
            height: this.gameHeight,
            scene: this.scene,
            parent: this.container,
            audio: {
                noAudio: true
            },
            input: {
                mouse: {
                    capture: this.interactive,
                    target: null
                }
            },
            callbacks: {
                postBoot: (game) => {
                    //game.scene.sleep('Game')
                    //game.loop.stop()
                    if (this.scene) {
                        this.scene.events.on("create", () => {
                            this.sceneCreated()
                        })
                    }
                }
            },
            scale: {
                mode: Phaser.Scale.FIT,
            }
        };
        this.game = new Phaser.Game(gameConfig);

    }

    drawState(state: GridworldState, animated: boolean = true) {
        this._stateToDraw = state;
        this._animated = animated
    }

    setAction(player_index: number, action: Direction) {
        const nextState = this.mdp.transition(this.state, action);
        this._stateToDraw = nextState;
        this.state = nextState
    }

    close() {
        this.game.renderer.destroy();
        this.game.loop.stop();
        this.game.destroy(true);
        this.game.canvas.remove();
    }

    sceneCreated() {
        if (this.displayTrajectory) {
            // TODO(nickswalker): Use states or SA pairs to specify trajectories
            const positions = []
            let state = this.mdp.getStartState()
            for (let i = 0; i < this.displayTrajectory.length; i++) {
                let nextState = this.mdp.transition(state, this.displayTrajectory[i])
                positions.push(nextState.agentPositions[0])
                state = nextState
            }
            this.scene._drawTrajectory(positions)
            this.scene._drawState(this.mdp.getStartState(), this.scene.sceneSprite, false)
            // Stop calls to update.
            this.scene.scene.pause()
        }
        // Wait for first draw
        this.scene.events.once("render", () => {
            if (this.sceneCreatedDelegate) {
                this.sceneCreatedDelegate()
            }
        })

    }
}

type SpriteMap = { [key: string]: any }

export class GridworldScene extends Phaser.Scene {
    gameParent: GridworldGame
    tileSize: number
    sceneSprite: object
    _interactive: boolean
    inputDelayTimeout: Phaser.Time.TimerEvent
    terrainMap: TerrainMap
    ANIMATION_DURATION = 300
    TIMESTEP_DURATION = 400

    constructor(gameParent: GridworldGame, terrainMap: TerrainMap, tileSize: number, interactive: boolean = true) {
        super({
            active: true,
            visible: true,
            key: 'Game',
        });
        this.terrainMap = terrainMap
        this.gameParent = gameParent
        this.tileSize = tileSize
        this.inputDelayTimeout = null
        this._interactive = interactive
    }

    set interactive(value: boolean) {

        this._interactive = value
    }

    preload() {
        this.load.atlas("colors",
            this.gameParent.assetsPath + "tiles_small.png",
            this.gameParent.assetsPath + "tiles_small.json");
        this.load.atlas("arrows",
            this.gameParent.assetsPath + "arrows.png",
            this.gameParent.assetsPath + "arrows.json");
        this.load.image("agent", this.gameParent.assetsPath + "robot.png")
        this.load.atlas("landmarks", this.gameParent.assetsPath + "landmarks.png", this.gameParent.assetsPath + "landmarks.json")
        this.interactive = this._interactive
    }

    create(data: object) {
        this.sceneSprite = {};
        this.drawLevel();
        this._drawState(this.gameParent.state, this.sceneSprite);
    }

    drawLevel() {
        //draw tiles
        let pos_dict = this.terrainMap
        for (let y = 0; y < pos_dict.length; y++) {
            for (let x = 0; x < pos_dict[0].length; x++) {
                const type = pos_dict[y][x]
                const [key, name] = terrain_to_img[type]
                // Landmarks need to sit on top of the default tile
                if (key === "landmarks") {
                    const [key, name] = terrain_to_img[TerrainType.Empty]
                    const tile = this.add.sprite(
                        this.tileSize * x,
                        this.tileSize * y,
                        key,
                        name
                    );
                    tile.setDisplaySize(this.tileSize, this.tileSize);
                    tile.setOrigin(0);
                    tile.setDepth(0)
                }
                const tile = this.add.sprite(
                    this.tileSize * x,
                    this.tileSize * y,
                    key,
                    name
                );
                tile.setDisplaySize(this.tileSize, this.tileSize);
                tile.setOrigin(0);
                if (key === "landmarks") {
                    tile.setDepth(2)
                }
            }
        }
    }

    _drawTrajectory(trajectory: Position[]) {
        const halfTile = this.tileSize * .5
        const path = new Phaser.Curves.Path(this.tileSize * 1 + halfTile, this.tileSize * (this.terrainMap.length - 1 - 1) + halfTile)
        const graphics = this.scene.scene.add.graphics()
        graphics.setDepth(1);
        graphics.lineStyle(this.tileSize / 4, 0x0000FF, 1.0)
        for (let i = 0; i < trajectory.length; i++) {
            const pos = trajectory[i]
            let [drawX, drawY] = [pos.x, this.terrainMap.length - pos.y - 1]
            path.lineTo(this.tileSize * drawX + halfTile, this.tileSize * drawY + halfTile)
        }
        const lastPos = trajectory[trajectory.length - 1]
        let [drawX, drawY] = [lastPos.x, this.terrainMap.length - lastPos.y - 1]
        const arrowHead = this.add.sprite(this.tileSize * drawX + halfTile, this.tileSize * drawY + halfTile, "arrows", "filled_head.png")
        arrowHead.setDisplaySize(this.tileSize, this.tileSize)
        arrowHead.setOrigin(.5)
        const secondLastPos = trajectory[trajectory.length - 2]
        const delta = [secondLastPos.x - lastPos.x, -1 * (secondLastPos.y - lastPos.y)]
        if (delta[0] == 1) {
            arrowHead.rotation = 1.57
        } else if (delta[1] == -1) {
            arrowHead.rotation = 3.14
        } else if (delta[0] == -1) {
            arrowHead.rotation = 4.71
        }
        arrowHead.setDepth(-1)
        path.draw(graphics)
        //graphics.generateTexture("trajectory")
    }

    _drawState(state: GridworldState, sprites: SpriteMap, animated: boolean = true) {
        // States are supposed to be fed at regular, spaced intervals, so no tweens are running usually.
        // We'll kill everything for situations where we are responding to a hard state override
        this.tweens.killAll()
        sprites = sprites ?? {}
        sprites["agents"] = sprites["agents"] ?? {}
        for (let p = 0; p < state.agentPositions.length; p++) {
            const agentPosition = state.agentPositions[p]
            // Flip Y to make lower-left the origin
            let [drawX, drawY] = [agentPosition.x, this.terrainMap.length - agentPosition.y - 1]
            if (typeof (sprites['agents'][p]) === 'undefined') {
                const agent = this.add.sprite(this.tileSize * drawX, this.tileSize * drawY, "agent");
                agent.setDisplaySize(this.tileSize, this.tileSize)
                agent.setOrigin(0)
                agent.setDepth(2)
                sprites['agents'][p] = agent;
            } else {
                const agent = sprites['agents'][p]
                if (animated) {
                    this.tweens.add({
                        targets: [agent],
                        x: this.tileSize * drawX,
                        y: this.tileSize * drawY,
                        duration: this.ANIMATION_DURATION,
                        ease: 'Linear',
                        onComplete: (tween, target, player) => {
                            target[0].setPosition(this.tileSize * drawX, this.tileSize * drawY);
                        }
                    })
                } else {
                    agent.setPosition(this.tileSize * drawX, this.tileSize * drawY);
                }
            }

        }
    }

    update(time: number, delta: number) {
        //console.log(this.scene.isPaused())
        // Blackout user controls for a bit while we animate the current step
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
        if (this.gameParent._stateToDraw) {
            let state = this.gameParent._stateToDraw;
            delete this.gameParent._stateToDraw;
            this._drawState(state, this.sceneSprite, this.gameParent._animated);
        }
    }
}

