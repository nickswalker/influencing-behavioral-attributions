import RuntimeError = WebAssembly.RuntimeError;

type Position = {x: number, y: number}

export enum Direction {
    NORTH,
    EAST,
    SOUTH,
    WEST
}
let [n, s, e, w] = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST];

const transition = new Map([[Direction.NORTH, [0, 1]], [Direction.SOUTH, [0, -1]], [Direction.EAST, [1, 0]], [Direction.WEST, [-1, 0]]])

export class GridworldState {
    agentPositions: Position[]
    constructor(agentPosition: Position[]) {
        this.agentPositions = agentPosition
    }

    deepcopy() {
        const clonedPositions = JSON.parse(JSON.stringify(this.agentPositions));
        return new GridworldState(clonedPositions)
    }

}

export enum TerrainType {
    Empty,
    Wall,
    Fire,
    Goal
}

export const characterToTerrainType: {[key: string]: TerrainType} = {"-": TerrainType.Empty, " ": TerrainType.Empty, "X": TerrainType.Wall, "F": TerrainType.Fire, "G": TerrainType.Goal}

export function textToTerrain(grid: string[]) {
    const asChar = grid.map((r) => [...r])
    let terrain: TerrainMap = []
    let playerPositions: {[key: number]: Position} = {}
    for (let y = 0; y < grid.length; y++) {
        terrain[y] = []
        for (let x = 0; x < grid[0].length; x++) {
            let c: string = asChar[y][x]
            const asNum = parseInt(c)
            if (typeof(asNum) !== "undefined" && !isNaN(asNum)) {
                if (playerPositions[asNum]) {
                    console.error("Duplicate player in grid: " + asNum + " at " + x + ", " + y)
                    throw new RuntimeError()
                }
                playerPositions[asNum] = {x: x, y:y}
                c = " "
            }
            const t = characterToTerrainType[c]
            if (typeof t == "undefined") {
                console.error("No terrain type for character: " + c)
                throw new RuntimeError()
            }
            terrain[y].push(t)
        }
    }

    return {terrain: terrain, playerPositions: playerPositions}
}

export type TerrainMap = TerrainType[][]

export class Gridworld {
    terrain: TerrainMap
    height: number
    width: number
    constructor(terrain: TerrainMap) {
        this.terrain = terrain
        this.width = this.terrain[0].length
        this.height = this.terrain.length
    }

    getStartState() {
        return new GridworldState([{x: 1, y: 1}])
    }

    transition(state: GridworldState, action: Direction) : GridworldState {
        const nextState = state.deepcopy();
        const curPosition = nextState.agentPositions[0];
        const [dx, dy] = transition.get(action);
        curPosition.x += dx;
        curPosition.y += dy;
        curPosition.x = Math.max(0, Math.min(curPosition.x, this.width))
        curPosition.y = Math.max(0, Math.min(curPosition.y, this.height))
        if (this.terrain[curPosition.y][curPosition.x] == TerrainType.Wall) {
            return state.deepcopy();
        }
        nextState.agentPositions[0] = curPosition
        return nextState
    }
}
