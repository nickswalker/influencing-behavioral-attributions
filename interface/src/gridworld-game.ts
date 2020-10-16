import "phaser"
import {Actions, GridMap, Gridworld, GridworldState, Position, TerrainType} from "./gridworld-mdp"
import KeyboardManager = Phaser.Input.Keyboard.KeyboardManager;
import CursorKeys = Phaser.Types.Input.Keyboard.CursorKeys;
import Key = Phaser.Input.Keyboard.Key;

const terrain_to_img: { [key in TerrainType]: string[] } = {
    [TerrainType.Empty]: ['colors', 'sol_base2.png'],
    [TerrainType.Goal]: ['landmarks', 'door.png'],
    [TerrainType.Fire]: ['landmarks', 'fire.png'],
    [TerrainType.Wall]: ['colors', "sol_base02_full.png"],
    [TerrainType.Reward]: ['landmarks', "treasure.png"],
};

export class GridworldGame {
    private gameWidth: number
    private gameHeight: number
    container: HTMLElement
    scene: any
    state: GridworldState
    assetsPath: string
    game: Phaser.Game
    _stateToDraw: GridworldState
    _rotationToDraw: number[]
    _animated: boolean
    displayTrajectory: GridworldState[]
    sceneCreatedDelegate?: () => void

    constructor(
        container: HTMLElement,
        tileSize = 32,
        assetsLoc = "assets/",
        mapName: string = null,
        terrain: GridMap = null
    ) {

        this.container = container;
        this.gameWidth = tileSize * 6
        this.gameHeight = tileSize * 6

        this.assetsPath = assetsLoc;
        this.scene = new GridworldScene(this, tileSize, mapName, terrain)
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
                keyboard: {
                    capture: []
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
                mode: Phaser.Scale.NONE,
            }
        };
        this.game = new Phaser.Game(gameConfig);

    }

    drawState(state: GridworldState, animated: boolean = true) {
        this._stateToDraw = state;
        this._animated = animated
    }

    rotateAgent(x: number, y: number) {
        this._rotationToDraw = [x, y]
    }

    setAction(player_index: number, action: Actions) {
        const nextState = this.scene.mdp.transition(this.scene.currentState, action);
        this._stateToDraw = nextState;
    }

    close() {
        this.game.renderer.destroy();
        this.game.loop.stop();
        this.game.destroy(true);
        this.game.canvas.remove();
    }

    sceneCreated() {
        if (this.displayTrajectory) {
            const positions = this.displayTrajectory.map((state) => {
                return state.agentPositions[0]
            })
            this.scene._drawTrajectory(positions)
            if (this.displayTrajectory.length > 0) {
                this.scene._drawState(this.displayTrajectory[0], this.scene.sceneSprite, false)
            }
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
    sceneSprite: SpriteMap
    mdp: Gridworld
    currentState: GridworldState
    mapName: string
    _interactive: boolean
    _trailGraphics: Phaser.GameObjects.Graphics
    inputDelayTimeout: Phaser.Time.TimerEvent
    ANIMATION_DURATION = 50
    map: any
    cursors: CursorKeys
    spacebar: Key

    constructor(gameParent: GridworldGame, tileSize: number, mapName: string = null, map: GridMap = null) {
        super({
            active: true,
            visible: true,
            key: 'Game',
        });
        this.gameParent = gameParent
        this.tileSize = tileSize
        this.inputDelayTimeout = null
        this._interactive = false
        this.mapName = mapName
        if (map) {
            this.mdp = new Gridworld(map.terrain)
        }
    }

    set interactive(value: boolean) {
        this._interactive = value
    }

    preload() {
        this.interactive = this._interactive
        // TODO(nickswalker): Fix handling of the hardcoded terrain case
        this.load.tilemapTiledJSON('map', this.gameParent.assetsPath + '/' + this.mapName + '.json');
        this.load.image('interior_tiles', this.gameParent.assetsPath + 'interior_tiles.png');
        this.load.image('agent', this.gameParent.assetsPath + 'roomba.png');
        this._trailGraphics = this.scene.scene.add.graphics()
    }

    create(data: object) {
        this.cursors = this.input.keyboard.createCursorKeys();
        this.spacebar = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.SPACE)
        this.sceneSprite = {};
        this.drawLevel();
        this.currentState = this.mdp.getStartState()
        this._drawState(this.currentState, this.sceneSprite);
    }

    drawLevel() {
        const map = this.make.tilemap({key: 'map'});
        this.game.scale.resize(map.width * map.tileWidth, map.height * map.tileHeight)
        const tileset = map.addTilesetImage('interior_tiles', 'interior_tiles');


        const ground = map.createStaticLayer('ground_walls', tileset, 0, 0);
        const deco0 = map.createStaticLayer('deco0', tileset, 0, 0);
        const deco1 = map.createStaticLayer('deco1', tileset, 0, 0);
        const deco2 = map.createStaticLayer('deco2', tileset, 0, 0);
        const roof = map.createStaticLayer('roof', tileset, 0, 0);
        if (map.getLayer("goal")) {
            const goal = map.createStaticLayer('goal', tileset, 0, 0)
        }
        const collision = map.getLayer("collision")
        const terrainMap = [...Array(map.height)].map(e => Array(map.width));
        for (let x = 0; x < map.width; x++) {
            for (let y = 0; y < map.height; y++) {
                const tile = map.getTileAt(x, y, false, "collision")
                if (tile) {
                    terrainMap[y][x] = TerrainType.Wall;
                }
            }
        }
        this.mdp = new Gridworld(terrainMap)
        this.map = map
    }

    _drawTrajectory(trajectory: Position[]) {
        if (trajectory.length == 0) {
            return;
        }
        const fT = this.tileSize
        const hT = this.tileSize * .5
        const startPos = trajectory[0]
        const path = new Phaser.Curves.Path(this.tileSize * startPos.x + hT, this.tileSize * startPos.y + hT)
        const graphics = this.scene.scene.add.graphics()
        graphics.setDepth(1);
        graphics.lineStyle(fT / 4, 0xa9cc29, 0.6)
        for (let i = 1; i < trajectory.length; i++) {
            const pos = trajectory[i]
            let [drawX, drawY] = [pos.x, pos.y]
            path.lineTo(fT * drawX + hT, fT * drawY + hT)
        }
        const lastPos = trajectory[trajectory.length - 1]
        let [drawX, drawY] = [lastPos.x, lastPos.y - 1]
        const arrowHead = this.add.sprite(fT * drawX + hT, fT * drawY + hT, "arrows", "filled_head.png")
        arrowHead.setDisplaySize(fT, fT)
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

    _drawRotation(rotation: number[], sprites: SpriteMap) {
        let angle = 0
        if (rotation[0] == -1) {
            angle = -90
        } else if (rotation[1] == -1) {
            angle = 0
        } else if (rotation[0] == 1) {
            angle = 90
        } else if (rotation[1] == 1) {
            angle = 180
        }
        this.tweens.add({
            targets: sprites["agents"][0],
            angle: angle,
            duration: this.ANIMATION_DURATION,
            onComplete: (tween, target, player) => {
                target[0].setAngle(angle);
            }
        })
    }

    _drawState(state: GridworldState, sprites: SpriteMap, animated: boolean = true) {
        // States are supposed to be fed at regular, spaced intervals, so no tweens are running usually.
        // We'll kill everything for situations where we are responding to a hard state override
        //this.tweens.killAll()
        const tS = this.tileSize;
        const hS = this.tileSize * 0.5;
        sprites = sprites ?? {}
        sprites["agents"] = sprites["agents"] ?? {}
        for (let p = 0; p < state.agentPositions.length; p++) {
            const agentPosition = state.agentPositions[p]
            // Flip Y to make lower-left the origin
            let [drawX, drawY] = [agentPosition.x, agentPosition.y]
            if (typeof (sprites['agents'][p]) === 'undefined') {
                const agent = this.add.sprite(tS * drawX + hS, tS * drawY + hS, "agent");
                agent.setDisplaySize(tS, tS)
                agent.setOrigin(0.5)
                agent.setDepth(2)
                sprites['agents'][p] = agent;
            } else {
                const agent = sprites['agents'][p]
                const currentX = agent.x
                const currentY = agent.y
                if (animated) {
                    /*const trail = this.add.sprite(currentX + hS, currentY + hS, "dot")
                    trail.setDisplaySize(tS, tS)
                    agent.setOrigin(0)
                    agent.setDepth(2)
                    this.tweens.add({
                        targets: trail,
                        opacity: 0,
                        ease: "Expo.easeOut",
                        duration: this.ANIMATION_DURATION * 400,
                        onComplete: (tween, target, player) => {
                            target[0].destroy()
                        }
                    })*/
                    this.tweens.add({
                        targets: agent,
                        x: tS * drawX + hS,
                        y: tS * drawY + hS,
                        duration: this.ANIMATION_DURATION,
                        onComplete: (tween, target, player) => {
                            target[0].setPosition(tS * drawX + hS, tS * drawY + hS);
                        }
                    })
                } else {
                    agent.setPosition(tS * drawX + hS, tS * drawY + hS);
                }
            }

        }
        this.events.emit("stateDrawn")
    }

    clearTrail() {
        this._trailGraphics.clear()
        this._trailGraphics.fillStyle(0xa9cc29)
        this._trailGraphics.setDepth(-1)
    }

    update(time: number, delta: number) {
        const agent = this.sceneSprite["agents"][0]
        if (this.tweens.isTweening(agent)) {
            const hT = .5 * this.tileSize
            this._trailGraphics.fillStyle(0xa9cc29)
            this._trailGraphics.setDepth(1)
            this._trailGraphics.fillRect(agent.x - .25 * hT, agent.y - .25 * hT, .5 * hT, .5 * hT)
        }
        //console.log(this.scene.isPaused())
        // Blackout user controls for a bit while we animate the current step
        if (this.inputDelayTimeout && this.inputDelayTimeout.getOverallProgress() == 1.0) {
            this.inputDelayTimeout = null
        }
        const acceptingInput = this.inputDelayTimeout == null
        if (this._interactive && acceptingInput) {
            let action: Actions = null
            if (this.cursors.left.isDown) {
                action = Actions.WEST
            } else if (this.cursors.right.isDown) {
                action = Actions.EAST
            } else if (this.cursors.up.isDown) {
                action = Actions.NORTH
            } else if (this.cursors.down.isDown) {
                action = Actions.SOUTH
            } else if (this.spacebar.isDown) {
                action = Actions.NONE
            }
            if (action !== null) {
                this.gameParent.setAction(0, action)
                this.inputDelayTimeout = this.time.addEvent({delay: 250})
            }
        }
        if (this.gameParent._stateToDraw) {
            let state = this.gameParent._stateToDraw;
            delete this.gameParent._stateToDraw;
            this.currentState = state
            this._drawState(state, this.sceneSprite, this.gameParent._animated);

        }
        if (this.gameParent._rotationToDraw) {
            let rotation = this.gameParent._rotationToDraw;
            delete this.gameParent._rotationToDraw;
            this._drawRotation(rotation, this.sceneSprite)
        }
    }
}

