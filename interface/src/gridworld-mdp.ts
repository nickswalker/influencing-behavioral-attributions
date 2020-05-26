import RuntimeError = WebAssembly.RuntimeError;

type Position = {x: number, y: number}

export enum Direction {
    NORTH,
    EAST,
    SOUTH,
    WEST
}

export class GridworldState {
    agentPosition: Position
    constructor(agentPosition: Position) {
        this.agentPosition = agentPosition
    }

    deepcopy() {
        return new GridworldState(this.agentPosition)
    }

}

export enum TerrainType {
    Empty,
    Wall,
    Fire,
    Goal
}

export const characterToTerrainType: {[key: string]: TerrainType} = {"-": TerrainType.Empty, " ": TerrainType.Empty, "X": TerrainType.Wall, "F": TerrainType.Fire, "O": TerrainType.Goal}

export function textToTerrain(grid: string[]) {
    const asChar = grid.map((r) => [...r])
    let terrain: TerrainMap = []
    let playerPositions: {[key: number]: Position} = {}
    for (let y = 0; y < grid.length; y++) {
        terrain[y] = []
        for (let x = 0; x < grid[0].length; x++) {
            let c: string = asChar[y][x]
            const asNum = Number(c)
            if (asNum) {
                if (playerPositions[asNum]) {
                    console.error("Duplicate player in grid")
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
    constructor(terrain: TerrainMap) {
        this.terrain = terrain
    }

    getStartState() {
        return new GridworldState({x: 0, y: 0})
    }
}
