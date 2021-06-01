question_code_to_name = {"b1": "broken", "b2": "clumsy", "b3": "competent", "b4": "confused", "b5": "curious",
                         "b6": "efficient", "b7": "energetic", "b8": "focused", "b9": "intelligent",
                         "b10": "investigative", "b11": "lazy", "b12": "lost", "b13": "reliable", "b14": "responsible", "b15": "inquisitive"}
old_question_names = sorted(list([question_code_to_name[f"b{i}"] for i in range(1,15)]))
short_question_names = sorted(list([question_code_to_name[f"b{i}"] for i in range(1,15)]))
short_question_names.remove("energetic")
short_question_names.remove("lazy")
question_names = sorted(list(question_code_to_name.values()))
feature_names = ["goal_cov",
                 "overlap",
                 "length",
                 "straight template match",
                 "hook template match",
                 "carpet time",
                 "collision time",
                 "redundant coverage",
                 "total coverage",
                 "idle time",
                 "start-stop template match"
                 ]
attributions = ["broken", "clumsy", "competent", "confused", "curious", "efficient", "energetic", "focused", "intelligent", "investigative", "lazy", "lost", "reliable", "responsible"]
cond_names = {0:"TSK", 1: "B+",2:"B-",3:"BAL", 4:"TSK", 5: "B+",6:"B-",7:"BAL", 8: "B+",9:"B-",10:"BAL"}
test_cond_names = {0:"TSK", 1: "B+",2:"B-",3:"BAL", 4: "B+",5:"B-",6:"BAL", 7: "B+",8:"B-",9:"BAL"}
factor_structure = {"competence":
                        ["competent", "efficient", "focused", "intelligent", "reliable", "responsible"],
                    "brokenness": ["broken", "clumsy", "confused", "lost"],
                    "curiosity": ["curious", "investigative"]
                    }