"""Generate benchmark/questions.jsonl. Questions/gold were drafted BEFORE any retrieval run.
Run from repo root: python benchmark/make_questions.py"""
import json
from pathlib import Path

T = {k: f"THE ADVENTURE OF {v}" for k, v in {
    "S01": "THE EMPTY HOUSE", "S03": "THE DANCING MEN", "S05": "THE PRIORY SCHOOL",
    "S08": "THE SIX NAPOLEONS", "S11": "THE MISSING THREE-QUARTER", "S12": "THE ABBEY GRANGE"}.items()}

# (id, split, type, question, expected_answer, unit, evidence_quote, filter_story_key or None)
Q = [
 ("D01","dev","direct","What weapon was used to shoot Ronald Adair through the open window?","An air-gun firing an expanding bullet.","S01-U011","shot the Honourable Ronald Adair with an expanding bullet from an air-gun",None),
 ("D02","dev","direct","Who constructed the air-gun that Holmes picked up from the floor, and for whom?","Von Herder, the blind German mechanic, for Professor Moriarty.","S01-U011","I knew Von Herder, the blind German mechanic, who constructed it to the order of the late Professor Moriarty.",None),
 ("D03","dev","paraphrase","On what charge does Holmes's young visitor expect to be arrested?","The murder of Mr. Jonas Oldacre of Lower Norwood.","S02-U002","Upon the charge of murdering Mr. Jonas Oldacre, of Lower Norwood.",None),
 ("D04","dev","direct","At which farm in Norfolk did Abe Slaney live?","Elriges Farm, East Ruston.","S03-U011","Mr. Abe Slaney, Elriges Farm, East Ruston, Norfolk.",None),
 ("D05","dev","paraphrase","What did Holmes's American correspondent reply when asked about Abe Slaney?","That he was the most dangerous crook in Chicago.","S03-U013","The most dangerous crook in Chicago.",None),
 ("D06","dev","direct","How was Captain Peter Carey killed?","A steel harpoon was driven through his chest, pinning him to the wall.","S06-U004","Right through his broad breast a steel harpoon had been driven",None),
 ("D07","dev","direct","Who did Holmes identify as the true murderer of Peter Carey?","Patrick Cairns.","S06-U012","Patrick Cairns, the true murderer of Peter Carey.",None),
 ("D08","dev","direct","How much money did Milverton demand from Lady Eva?","Seven thousand pounds.","S07-U002","Seven thousand pounds.",None),
 ("D09","dev","paraphrase","What valuable object was hidden inside the bust of Napoleon?","The famous black pearl of the Borgias.","S08-U011","the famous black pearl of the Borgias",None),
 ("D10","dev","direct","What text made up the examination paper that arrived from the printers?","Half a chapter of Thucydides.","S09-U002","The exercise consists of half a chapter of Thucydides.",None),
 ("D11","dev","conditional","What did the student decide to do instead of sitting the examination after he gave up his unfair advantage?","Accept a commission in the Rhodesian Police and go to South Africa.","S09-U011","I have been offered a commission in the Rhodesian Police",None),
 ("D12","dev","metadata-filter","Who is the client that consults Holmes in this story?","Mr. Hilton Cubitt of Riding Thorpe Manor, Norfolk.","S03-U001","Mr. Hilton Cubitt, of Riding Thorpe Manor, Norfolk",  "S03"),
 ("D13","dev","metadata-filter","Who is the client that consults Holmes in this story?","Dr. Thorneycroft Huxtable, head of the Priory School.","S05-U001","the first appearance of Thorneycroft Huxtable, M.A., Ph.D., etc.","S05"),
 ("D14","dev","metadata-filter","Which police inspector is handling the case and meets Holmes at the door?","Inspector Stanley Hopkins.","S12-U002","Inspector Stanley Hopkins confronted us in the open doorway.","S12"),
 ("D15","dev","metadata-filter","Which Scotland Yard detective brings the case to Holmes?","Mr. Lestrade.","S08-U001","Mr. Lestrade, of Scotland Yard, to look in upon us of an evening","S08"),
 ("T01","test","metadata-filter","Who comes to Baker Street to ask for Holmes's help with the case?","Mr. Cyril Overton of Trinity College, Cambridge.","S11-U001","the card of Mr. Cyril Overton, Trinity College, Cambridge","S11"),
 ("T02","test","conditional","What observation about the professor's food gave Holmes a clue that a second person was hidden in the house?","His consumption of food had increased, as expected when supplying a second person.","S10-U015","as one would expect when he is supplying a second person",None),
 ("T03","test","direct","What important piece of evidence was found clasped in the dead man's hand?","A golden pince-nez with two broken ends of black silk cord.","S10-U005","a golden pince-nez, with two broken ends of black silk cord dangling from",None),
 ("T04","test","paraphrase","Who is the Premier who visits Holmes together with Trelawney Hope?","Lord Bellinger, twice Premier of Britain.","S13-U001","the illustrious Lord Bellinger, twice Premier of Britain",None),
 ("T05","test","direct","How large a reward did the Duke offer to whoever could tell him where his son was?","A check for five thousand pounds.","S05-U002","a check for five thousand pounds will be handed over to the person who can tell him where his son is",None),
]
rows = []
for id_, split, typ, q, a, unit, quote, f in Q:
    doc, u = unit.split("-U")
    rows.append({"id": id_, "split": split, "type": typ, "question": q, "expected_answer": a,
                 "gold_source_id": doc, "gold_page": int(u), "gold_unit_id": unit,
                 "evidence_quote": quote, "filter": {"story": T[f]} if f else None})
out = Path(__file__).parent / "questions.jsonl"
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
print(len(rows), "questions written")
