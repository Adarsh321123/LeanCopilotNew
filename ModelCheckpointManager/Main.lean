import ModelCheckpointManager.Url
import ModelCheckpointManager.Download
import Init.System.IO
import Lean
import LeanCopilot.Models.Builtin
import LeanCopilot.Models.Native
import LeanCopilot.Models.Registry
import Batteries.Data.HashMap

open Lean
open LeanCopilot
open Batteries

-- TODO: change?
def configPath : IO System.FilePath := do
  let home ← IO.getEnv "HOME"
  pure $ (home.getD "/") / ".lean_copilot_current_model"

def saveCurrentModel (model : String) : IO Unit := do
  let config ← configPath
  IO.FS.writeFile config model

def loadCurrentModel : IO String := do
  let config ← configPath
  if ← config.pathExists then
    let contents ← IO.FS.readFile config
    pure contents.trim
  else
    pure Builtin.generator.name

-- TODO: do we need this if we can just download a new url immediately? when we restart the file, this won't be remembered anyway, right?
-- Define a mutable reference to store additional URLs
initialize additionalModelUrlsRef : IO.Ref (List String) ← IO.mkRef []

-- initialize currentModelRef : IO.Ref String ← IO.mkRef Builtin.generator.name

initialize currentModelRef : IO.Ref String ← do
  let model ← loadCurrentModel
  IO.mkRef model

def builtinModelUrls : List String := [
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-tacgen-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/premise-embeddings-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-byt5-small"
]

-- Function to get all model URLs (built-in + additional)
def getAllModelUrls : IO (List String) := do
  let additional ← additionalModelUrlsRef.get
  return builtinModelUrls ++ additional

def getCurrentModel : IO String := currentModelRef.get

def saveRegisteredGenerators (generators : Batteries.HashMap String Generator) : IO Unit := do
  let config ← configPath
  IO.println s!"Saving registered generators to {config}.generators"
  -- let generatorNames := generators.toList.map (fun (name, _) => name)
  let generatorUrls := generators.toList.map fun (name, gen) =>
    match gen with
    | .native ng => ng.url.toString
    | _ => ""  -- Handle other generator types if needed
  -- IO.println s!"Generator names: {generatorNames}"
  IO.println s!"Generator URLs: {generatorUrls}"
  IO.FS.writeFile (toString config ++ ".generators") (String.intercalate "\n" generatorUrls)
  IO.println "Saved registered generators to file"

def loadAndRegisterGenerators : IO Unit := do
  let config ← configPath
  IO.println s!"Loading registered generators from {config}.generators"
  let generatorsFile := System.FilePath.mk $ toString config ++ ".generators"
  IO.println s!"Generators file: {generatorsFile}"
  if ← generatorsFile.pathExists then
    IO.println "Generators file exists"
    let contents ← IO.FS.readFile generatorsFile
    IO.println s!"Contents: {contents}"
    -- let generatorNames := contents.trim.split (· == '\n')
    -- IO.println s!"Generator names: {generatorNames}"
    let generatorUrls := contents.trim.split (· == '\n')
    IO.println s!"Generator URLs: {generatorUrls}"
    -- for name in generatorNames do
    for url in generatorUrls do
      IO.println s!"Loading generator for URL: {url}"
      let parsedUrl := Url.parse! url
      let newGenerator : NativeGenerator := {
        url := parsedUrl
        tokenizer := ByT5.tokenizer
        params := { numReturnSequences := 32 }
      }
      registerGenerator parsedUrl.name! (.native newGenerator)
      IO.println s!"Registered generator with name {parsedUrl.name!}"

-- Function to add a new URL to the additional URLs list
def addModelUrl (url : String) : IO Unit := do
  IO.println "Adding new generator url"
  additionalModelUrlsRef.modify (url :: ·)
  let url := Url.parse! url

  IO.println "Updating current model and saving to file"
  -- Re-register the option with the new URL
  -- Lean.registerOption `LeanCopilot.suggest_tactics.model {
  --   defValue := url.name!
  -- }
  currentModelRef.set url.name!
  saveCurrentModel url.name!

  IO.println "Registering new generator"
  let newGenerator : NativeGenerator := {
    url := url
    tokenizer := ByT5.tokenizer
    params := {
      numReturnSequences := 32
    }
  }
  IO.println "Calling register generator"
  registerGenerator url.name! (.native newGenerator)

  -- Save the updated list of registered generators
  let mr ← getModelRegistry
  IO.println "got model registry"
  saveRegisteredGenerators mr.generators
  IO.println "saved registered generators"

  -- TODO: reduce duplication
  IO.println "Downloading new generator"
  let mut tasks := #[]
  let urls ← getAllModelUrls
  let parsedUrls := Url.parse! <$> urls

  for url in parsedUrls do
    tasks := tasks.push $ ← IO.asTask $ downloadUnlessUpToDate url

  for t in tasks do
    match ← IO.wait t with
    | Except.error e => throw e
    | Except.ok _ => pure ()


structure Request where
  text : String
deriving ToJson

structure Response where
  output : String
deriving FromJson

def sendStartTraining {α β : Type} [ToJson α] [FromJson β] (req : α) (url : String) : IO β := do
  let reqStr := (toJson req).pretty 99999999999999999
  IO.println s!"Sending request: {reqStr}"
  let out ← IO.Process.output {
    cmd := "curl"
    args := #["-X", "POST", url, "-H", "accept: application/json", "-H", "Content-Type: application/json", "-d", reqStr]
  }
  IO.println s!"Raw output: {out.stdout}"
  if out.exitCode != 0 then
     throw $ IO.userError s!"Request failed. Please check if the server is up at `{url}`."
  let some json := Json.parse out.stdout |>.toOption
    | throw $ IO.userError "Failed to parse response 1"
  let some res := (fromJson? json : Except String β) |>.toOption
    | throw $ IO.userError "Failed to parse response 2"
  return res

def main (args : List String) : IO Unit := do
  let mut tasks := #[]
  let urls ← if args.isEmpty then getAllModelUrls else pure args
  let parsedUrls := Url.parse! <$> urls

  for url in parsedUrls do
    tasks := tasks.push $ ← IO.asTask $ downloadUnlessUpToDate url

  for t in tasks do
    match ← IO.wait t with
    | Except.error e => throw e
    | Except.ok _ => pure ()

  -- TODO: some of these urls like the premise retriever should not be registered
  loadAndRegisterGenerators

  -- Start progressive training with the initial repository
  -- TODO: ask for url somehow
  IO.println "Starting the program"
  let url := "http://127.0.0.1:8000/reverse/"
  let req : Request := {
    text := "hello"
  }
  -- TODO: make sure the request is sent only once or any request after that is avoided
  let res : Response ← sendStartTraining req url
  IO.println s!"Final response: {res.output}"

  println! "Done!"
