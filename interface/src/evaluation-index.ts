import {setUpPages} from "./paged-navigation.js";
import {shuffleArray} from "./utils.js";
import "./gridworld-interactive"
import "./gridworld-trajectory-player"

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

}

let debugMode = false

function render(trajectoryIds: number[]) {
    let attributionInsertionContainer = document.querySelector("#attribution-insertion-region")
    let conditionTemplate = document.querySelector("#attribution").innerHTML;
    let questionsTemplate = document.querySelector("#questions").innerHTML;

    for (let i = 0; i < trajectoryIds.length; i++) {
        let condition = Mustache.render(conditionTemplate,
            {"trajectoryId": String(trajectoryIds[i]),
                "condition": String(i),
                "robotNumber": String(i + 1),
                "required":"required",
                "unwatched": "unwatched"})
        attributionInsertionContainer.innerHTML += condition;
        let questions = Mustache.render(questionsTemplate,{"trajectoryId": String(trajectoryIds[i]),
            "condition": String(i),
            "robotNumber": String(i + 1),
            "required":"required"})
        attributionInsertionContainer.querySelector(".page:last-of-type .questions").innerHTML += questions
    }
    let warmupQuestions = Mustache.render(questionsTemplate,{
        "condition": "9999",
        "robotNumber": ""})
    document.querySelector(".warm-up .questions").innerHTML += warmupQuestions

}


function addVideoListeners() {
    document.querySelectorAll(".video-container").forEach((container: any) =>{
        let video = container.querySelector("video")

        // @ts-ignore
        let plyr = new Plyr(video, {"displayDuration": false,
            tooltips: { controls: true, seek: false},
            controls: ['play-large', 'play', 'progress', 'restart'],
            listeners: {
                seek: (e:any) => {
                    if (!(plyr.media.classList.contains("requires-finished-to-seek") && plyr.media.classList.contains("unwatched"))){
                        return true;
                    }
                    e.preventDefault();
                    console.log(`prevented`);
                    return false;
                }
            }
        });
        plyr.elements.container.classList += " " + plyr.media.classList

        plyr.once("ended", (e:any)=>{
            plyr.media.classList.remove("unwatched")
            plyr.elements.container.classList.remove("unwatched")
        })
    })
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
    if (debugMode) {
        console.log("Skipping validation")
        return null;
    }
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

    } else if (page.classList.contains("warm-up")) {
        let currentVideo = page.querySelector("video");
        if (currentVideo && currentVideo.classList.contains("unwatched")) {
            return "Please watch the full video before continuing";

        }
    }  else if (page.classList.contains("attribution")) {
        let currentVideo = page.querySelector("video");
        if (currentVideo) {
            if (currentVideo.classList.contains("unwatched")) {
                return "Please watch the full video before continuing";
            }
        }
        let radioGroups = page.querySelectorAll("fieldset");
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
        const missingDescribe = Array.from(page.querySelectorAll<any>("crowd-input[name$=describe]")).filter((x)=>{return x.required && x.value.length < 10})
        const missingExplain = Array.from(page.querySelectorAll<any>("crowd-input[name$=explain]")).filter((x)=>{return x.required && x.value.length < 10})
        formValid = formValid && missingDescribe.length === 0 && missingExplain.length === 0
        if (!formValid) {
            return "Please respond to all required questions before continuing";
        }
    } else if (page.classList.contains("comparison")) {
        let radioGroups = page.querySelectorAll("fieldset");
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
        const missingExplain = Array.from(page.querySelectorAll<any>("crowd-input[name$=explain]")).filter((x)=>{return x.required && x.value.length < 10})
        formValid = formValid && missingExplain.length === 0
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


function addFinishedListeners(){
    document.querySelectorAll("fieldset input[type=radio]").forEach((radioButton: Element) => {
        radioButton.addEventListener("change",()=>
        {
            radioButton.parentElement.classList.add("finished")
        })
    })

    document.querySelectorAll<HTMLInputElement>("crowd-input").forEach((input: HTMLInputElement) => {
        input.addEventListener("keyup", ()=>{
            if (!input.classList.contains("min-length-10")){
                return true;
            }
            if (input.value.length >= 10 ) {
                input.classList.add("finished")
            } else {
                input.classList.remove("finished")
            }
        })
    })

}

document.addEventListener('keydown', function(event) {
    if (event.ctrlKey && event.key === 'p') {
        debugMode = !debugMode
        if (debugMode) {
            alert('Debug mode enabled');
        } else {
            alert('Debug mode disabled')
        }
    }
});


render(window.trajectoryIds)
let questionOrdering = randomizeQuestionOrder()
addVideoListeners()

let questionOrderingInput = document.querySelector<HTMLInputElement>("input[name=questionOrdering]")
questionOrderingInput.value = questionOrdering[0].toString()
nameAllInputs()
setUpPages(document.getElementById("page-container"), validateNext)

addFinishedListeners()