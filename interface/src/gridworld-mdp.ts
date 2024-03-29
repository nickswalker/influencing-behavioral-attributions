export type Position = { x: number, y: number }

export enum Actions {
    NONE,
    EAST,
    NORTH,
    WEST,
    SOUTH
}

let [n, s, e, w] = [Actions.NORTH, Actions.SOUTH, Actions.EAST, Actions.WEST];

const transition = new Map([[Actions.NONE, [0, 0]], [Actions.NORTH, [0, -1]], [Actions.SOUTH, [0, 1]], [Actions.EAST, [1, 0]], [Actions.WEST, [-1, 0]]])

export class GridworldState {
    agentPositions: Position[]

    constructor(agentPosition: Position[]) {
        this.agentPositions = agentPosition
    }

    deepcopy() {
        const clonedPositions = JSON.parse(JSON.stringify(this.agentPositions));
        return new GridworldState(clonedPositions)
    }

    equals(other: GridworldState) {
        return this.agentPositions[0].x == other.agentPositions[0].x && this.agentPositions[0].y == other.agentPositions[0].y
    }

}

export enum TerrainType {
    Empty,
    Wall,
    Fire,
    Goal,
    Reward
}

export const characterToTerrainType: { [key: string]: TerrainType } = {
    "-": TerrainType.Empty,
    " ": TerrainType.Empty,
    "X": TerrainType.Wall,
    "F": TerrainType.Fire,
    "G": TerrainType.Goal,
    "T": TerrainType.Reward,
    "R": TerrainType.Reward,
}

export type TerrainMap = TerrainType[][]

export type GridMap = { terrain: TerrainMap, playerPositions: { [key: number]: Position } }

export class Gridworld {
    terrain: TerrainMap
    height: number
    width: number
    private startState: GridworldState
    private terminalState: GridworldState

    constructor(terrain: TerrainMap, startState: GridworldState,
                terminalState: GridworldState = null) {
        this.terrain = terrain
        this.width = this.terrain[0].length
        this.height = this.terrain.length
        this.startState = startState
        if (terminalState === null) {
            terminalState = startState
        }
        this.terminalState = terminalState
    }

    getStartState() {
        return this.startState.deepcopy()
    }

    getTerminalState() {
        return this.terminalState.deepcopy()
    }

    transition(state: GridworldState, action: Actions): GridworldState {
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
