import {characterToTerrainType, GridMap, GridworldState, Position, TerrainMap} from "./gridworld-mdp";

export function shuffleArray(array: any[]) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

export function textToStates(states: string) {
    const trajStates = JSON.parse(states
        .replace(/\(/g, '[')
        .replace(/\)/g, ']')
    );
    return trajStates.map((value: number[]) => {
        return new GridworldState([{x: value[0], y: value[1]}])
    })
}

export function statesToText(states: GridworldState[]) {
    let out = "["
    const tupleStrings = states.map((state) => {
        let pos = state.agentPositions[0]
        return "(" + pos.x + ", " + pos.y + ")"
    })
    for (let i = 0; i < tupleStrings.length; i++) {
        if (i > 0) {
            out += ","
        }
        out += tupleStrings[i]
    }
    out += "]"
    return out;
}

export function textToTerrain(grid: string[]): GridMap {
    const asChar = grid.map((r) => [...r])
    let terrain: TerrainMap = []
    let playerPositions: { [key: number]: Position } = {}
    for (let y = 0; y < grid.length; y++) {
        terrain[y] = []
        for (let x = 0; x < grid[0].length; x++) {
            let c: string = asChar[y][x]
            const asNum = parseInt(c)
            if (typeof (asNum) !== "undefined" && !isNaN(asNum)) {
                if (playerPositions[asNum]) {
                    console.error("Duplicate player in grid: " + asNum + " at " + x + ", " + y)
                    throw new Error()
                }
                playerPositions[asNum] = {x: x, y: y}
                c = " "
            }
            const t = characterToTerrainType[c]
            if (typeof t == "undefined") {
                console.error("No terrain type for character: " + c)
                throw new Error()
            }
            terrain[y].push(t)
        }
    }

    return {terrain: terrain, playerPositions: playerPositions}
}

export function hashCode(str: string) {
    return str.split('').reduce((prevHash, currVal) =>
        (((prevHash << 5) - prevHash) + currVal.charCodeAt(0))|0, 0);
}