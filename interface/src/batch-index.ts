import {setUpPages} from "./paged-navigation.js";
import {shuffleArray} from "./utils.js";
import "./gridworld-interactive"

declare global {
    interface Mustache {

    }
    interface Plyr {
        setup(selector: any, options:any): any
    }
    interface Math {
        seedrandom(seed: number): any
    }

    let Plyr: Plyr;
    var trajectoryIds: number[];
    var attributions: string[];

}

function render(attributions: string[], trajectoryIds: number[]) {
    let attributionInsertionContainer = document.querySelector("#attribution-insertion-region")
    let conditionTemplate = document.querySelector("#attribution").innerHTML;

    for (let i = 0; i < trajectoryIds.length; i++) {
        let condition = Mustache.render(conditionTemplate, {"trajectoryId": String(trajectoryIds[i]), "condition": String(i), "robotNumber": String(i + 1), "required":"required", "unwatched": "unwatched"})
        attributionInsertionContainer.innerHTML += condition;
    }

    let demonstrationInsertionContainer = document.querySelector("#demonstration-insertion-region")
    let demonstrationTemplate = document.querySelector("#demonstration").innerHTML;

    for (let i = 0; i < attributions.length; i++) {
        let condition = Mustache.render(demonstrationTemplate, {"attribution": String(attributions[i]), "condition": String(i), "robotNumber": String(i + 1), "required":"required"})
        demonstrationInsertionContainer.innerHTML += condition;
    }

}

function addVideoListeners() {
    let allVideoSections = document.querySelectorAll(".video-container")
    for (let i = 0; i < allVideoSections.length; i++) {
        let videoSection = allVideoSections[i];
        let video = videoSection.querySelector("video");
        let restartControl = videoSection.querySelector(".video-controls .restart")
        restartControl.addEventListener("click", function(){
            video.currentTime = 0
        })
        video.addEventListener("ended", function(){
            video.classList.remove("unwatched")
        })
    }
}

function nameAllInputs() {
    // Silence any errors by marking no-name inputs as noops
    // MTurk will complain about unnamed inputs
    let allInputs = document.getElementsByTagName("input")
    for (let i = 0; i < allInputs.length; i++) {
        let input = allInputs[i]
        if (input.name) {
            continue;
        }
        input.name = "noop"+ String(i)
    }
}


function validateNext(page: HTMLElement) : string {
    if (page.classList.contains("qualification")) {
        let radioGroup = page.querySelector("crowd-radio-group");
        let options = radioGroup.querySelectorAll("crowd-radio-button")
        let rowValid = false;
        for (let i = 0; i < options.length; i++) {
            let option: HTMLInputElement = <HTMLInputElement>options[i]
            if (option.checked) rowValid = true;
            if (i === 0 && option.checked) {
                return "You are not eligible for this HIT. Please return it."
            }
        }
        if (!rowValid){
            return "Please select an answer"
        }

    } else if (page.classList.contains("attribution")) {
        let currentVideo = page.querySelector("video");
        if (currentVideo) {
            if (currentVideo.classList.contains("unwatched")) {
                return "Please watch the full video before continuing";
            }
        }
        let radioGroups = page.querySelectorAll("crowd-radio-group");
        let formValid = true;
        for (let i = 0; i < radioGroups.length; i++) {
            let radioGroup = radioGroups[i]
            let options = <any>radioGroup.querySelectorAll("input[required]")
            let rowValid = false;
            if (options.length == 0) {
                // Must not be any required items
                rowValid = true;
            } else {
                for (let j = 0; j < options.length; j++) {
                    if (options[j].checked) rowValid = true;
                }
            }
            if (!rowValid){
                formValid = false;
                break;
            }
        }
        // Make sure descriptions are actually sentence length
        let requiredInputs = <any>page.querySelectorAll("crowd-input[name$=describe]")
        for (let i = 0; i < requiredInputs.length; i++) {
            let input = requiredInputs[i]
            if (input.required && input.value.length < 10) {
                formValid = false;
            }
        }

        if (!formValid) {
            return "Please respond to all required questions before continuing";
        }
    }
    return null
}

function randomizeQuestionOrder(){
    // Apply randomization (Knuth-Fisher-Yates)
    // Note: NodeLists (return type of querySelector) are not shuffleable in place, thus the strange child append dance
    const questionSections = document.querySelectorAll('[class *= questions]');
    let seed = Math.random();

    for (let i = 0; i < questionSections.length; i++) {
        let questionSection = questionSections[i];
        // Randomize question order
        let questions = questionSection.querySelector('.question-randomization-region')
        // We use the same random sequence for each condition. Bump this rng out of the loop
        // if you want it to be random for each section
        Math.seedrandom(seed);
        for (let j = questions.children.length; j >= 0; j--) {
            let index = Math.floor(Math.random() * j);
            questions.appendChild(questions.children[index]);
        }
    }
    let allOrderings = []
    for (let i = 0; i < questionSections.length; i++) {
        let sectionOrdering = []
        let questionSection = questionSections[i];
        // Randomize question order
        let questions = questionSection.querySelector('.question-randomization-region')
        for (let j = 0;  j < questions.children.length; j++) {
            let questionName = questions.children[j].className
            sectionOrdering.push(questionName)
        }
        allOrderings.push(sectionOrdering)
    }


    return allOrderings
}


render(window.attributions,window.trajectoryIds)
randomizeQuestionOrder()
Plyr.setup('video', {"displayDuration": false, tooltips: { controls: false, seek: false}, controls: ['play-large', 'play', 'progress']});
addVideoListeners()
nameAllInputs()
setUpPages(document.getElementById("task"), validateNext)