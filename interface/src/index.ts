import "./gridworld-mdp"
import "./gridworld-game"
import {Direction, TerrainMap, textToTerrain} from "./gridworld-mdp";
import {GridworldGame} from "./gridworld-game";
import * as io from "socket.io-client"

let [n, s, e, w] = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST];

let terrain: {terrain: TerrainMap} = textToTerrain([
    'XXXXX',
    'O   O',
    '-----',
    '1----',
    'XXXXX'
])
let game = new GridworldGame(terrain["terrain"],
    document.getElementById("task")
)

game.init();
let atraj = [n, s, e, w]
let state = game.mdp.getStartState();

const socket = io("http://localhost:5000");
socket.emit("message", "HELLO WORLD");
socket.on('connect', function() {
    console.log("sent event")
    socket.emit('my event', {data: 'I\'m connected!'});
});
