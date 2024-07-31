import Lean
import LeanCopilot.Options
import LeanCopilot.Frontend
import Aesop.Util.Basic
import Batteries.Data.String.Basic
import Init.System.IO
import LeanCopilot.Models.Native
import LeanCopilot.Models.Registry
import LeanCopilot.Models.Builtin
import LeanCopilot.Models.ByT5
import ModelCheckpointManager.Main

open Lean Meta Parser Elab Term Tactic


set_option autoImplicit false


namespace LeanCopilot

/--
Pretty-print a list of goals.
-/
def ppTacticState : List MVarId → MetaM String
  | [] => return "no goals"
  | [g] => return (← Meta.ppGoal g).pretty
  | goals =>
      return (← goals.foldlM (init := "") (fun a b => do return s!"{a}\n\n{(← Meta.ppGoal b).pretty}")).trim


/--
Pretty-print the current tactic state.
-/
def getPpTacticState : TacticM String := do
  let goals ← getUnsolvedGoals
  ppTacticState goals


open SuggestTactics in
/--
Generate a list of tactic suggestions.
-/
def suggestTactics (targetPrefix : String) : TacticM (Array (String × Float)) := do
  IO.println s!"Inside suggestTactics"
  let state ← getPpTacticState
  let nm ← getCurrentGeneratorNameIO
  IO.println s!"Current generator name: {nm}"
  let model ← getGenerator nm
  -- let suggestions ← generate model state targetPrefix
  -- -- A temporary workaround to prevent the tactic from using the current theorem.
  -- -- TODO: Use a more principled way, e.g., see `Lean4Repl.lean` in `LeanDojo`.
  -- if let some declName ← getDeclName? then
  --   let theoremName := match declName.toString with
  --     | "_example" => ""
  --     | n => n.splitOn "." |>.getLast!
  --   let theoremNameMatcher := String.Matcher.ofString theoremName
  --   if ← isVerbose then
  --     logInfo s!"State:\n{state}"
  --     logInfo s!"Theorem name:\n{theoremName}"
  --   let filteredSuggestions := suggestions.filterMap fun ((t, s) : String × Float) =>
  --     let isAesop := t == "aesop"
  --     let isSelfReference := ¬ (theoremName == "") ∧ (theoremNameMatcher.find? t |>.isSome)
  --     if isSelfReference ∨ isAesop then none else some (t, s)
  --   IO.println "About to return filtered suggestions inside if"
  --   IO.println filteredSuggestions
  --   return filteredSuggestions
  -- else
  --   let filteredSuggestions := suggestions.filterMap fun ((t, s) : String × Float) =>
  --     let isAesop := t == "aesop"
  --     if isAesop then none else some (t, s)
  --   IO.println "About to return filtered suggestions outside if"
  --   IO.println filteredSuggestions
  --   return filteredSuggestions
  return #[]


/--
Information of a premise.
-/
structure PremiseInfo where
  name : String
  path : String
  code : String
  score : Float


/--
Annotate a premise with its type, doc string, import module path, and definition code.
-/
private def annotatePremise (pi : PremiseInfo) : MetaM String := do
  let declName := pi.name.toName
  try
    let info ← getConstInfo declName
    let premise_type ← Meta.ppExpr info.type
    let some doc_str ← findDocString? (← getEnv) declName
      | return s!"\n{pi.name} : {premise_type}"
    return s!"\n{pi.name} : {premise_type}\n```doc\n{doc_str}```"
  catch _ => return s!"\n{pi.name} needs to be imported from {pi.path}.\n```code\n{pi.code}\n```"


/--
Retrieve a list of premises given a query.
-/
def retrieve (input : String) : TacticM (Array PremiseInfo) := do
  if ¬ (← premiseEmbeddingsInitialized) ∧ ¬ (← initPremiseEmbeddings .auto) then
    throwError "Cannot initialize premise embeddings"

  if ¬ (← premiseDictionaryInitialized) ∧ ¬ (← initPremiseDictionary) then
    throwError "Cannot initialize premise dictionary"

  let k ← SelectPremises.getNumPremises
  let query ← encode Builtin.encoder input

  let rawPremiseInfo := FFI.retrieve query k.toUInt64
  let premiseInfo : Array PremiseInfo := rawPremiseInfo.map fun (name, path, code, score) =>
    { name := name, path := path, code := code, score := score }
  return premiseInfo


/--
Retrieve a list of premises using the current pretty-printed tactic state as the query.
-/
def selectPremises : TacticM (Array PremiseInfo) := do
  retrieve (← getPpTacticState)

structure StatusResponse where
  completed : Bool
deriving FromJson

def get (url : String) : IO StatusResponse := do
  let out ← IO.Process.output {
    cmd := "curl"
    args := #["-X", "GET", url, "-H", "accept: application/json", "-H", "Content-Type: application/json"]
  }
  IO.println s!"Raw output: {out.stdout}"
  if out.exitCode != 0 then
     throw $ IO.userError s!"Request failed. Please check if the server is up at `{url}`."
  let some json := Json.parse out.stdout |>.toOption
    | throw $ IO.userError "Failed to parse response 1"
  let some res := (fromJson? json : Except String StatusResponse) |>.toOption
    | throw $ IO.userError "Failed to parse response 2"
  return res

def newModelUrl := "https://huggingface.co/kaiyuy/leandojo-lean4-tacgen-byt5-small"

syntax "pp_state" : tactic
syntax "suggest_tactics" : tactic
syntax "suggest_tactics" str : tactic
syntax "select_premises" : tactic


macro_rules
  | `(tactic | suggest_tactics%$tac) => `(tactic | suggest_tactics%$tac "")


elab_rules : tactic
  | `(tactic | pp_state) => do
    let state ← getPpTacticState
    logInfo state

  | `(tactic | suggest_tactics%$tac $pfx:str) => do
    IO.println s!"Inside suggest_tactics"
    IO.println s!"Prefix: {pfx.getString}"

    -- Check the status of progressive training
    -- TODO: make new function
    -- TODO: do this for proof and retrieve premise too
    IO.println "Asking for status"
    let url := "http://127.0.0.1:8000/check-status/"
    let result ← get url
    IO.println s!"API call result: {result.completed}"
    -- If the status is completed, update the model LeanCopilot uses
    -- TODO: uncomment
    -- if result.completed then
    IO.println "Status completed. Using new model..."
    addModelUrl newModelUrl
    IO.println "Registering new generator"
    -- TODO: only do this one time though
    loadAndRegisterGenerators
    IO.println "Added new model"

    -- TODO: must call lake exe LeanCopilot/download to download the new model before suggesitng


    let (tacticsWithScores, elapsed) ← Aesop.time $ suggestTactics pfx.getString
    IO.println s!"Elapsed time: {elapsed.printAsMillis}"
    -- if ← isVerbose then
    --   logInfo s!"{elapsed.printAsMillis} for generating {tacticsWithScores.size} tactics"
    -- let tactics := tacticsWithScores.map (·.1)
    -- IO.println tactics
    -- if ← isVerbose then
    --   logInfo s!"Tactics: {tactics}"
    -- let range : String.Range := { start := tac.getRange?.get!.start, stop := pfx.raw.getRange?.get!.stop }
    -- IO.println "Got range as string"
    -- let ref := Syntax.ofRange range
    -- IO.println "Got range as syntax"
    -- hint ref tactics (← SuggestTactics.checkTactics)
    -- IO.println "Hinted"

  | `(tactic | select_premises) => do
    let premisesWithInfoAndScores ← selectPremises
    let rankedPremisesWithInfoAndScores := premisesWithInfoAndScores.qsort (·.score > ·.score)
    let richPremises ← Meta.liftMetaM $ (rankedPremisesWithInfoAndScores.mapM annotatePremise)
    let richPremisesExpand := richPremises.foldl (init := "") (· ++ · ++ "\n")
    logInfo richPremisesExpand


end LeanCopilot
