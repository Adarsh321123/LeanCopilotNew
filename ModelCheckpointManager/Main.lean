import ModelCheckpointManager.Url
import ModelCheckpointManager.Download
import Init.System.IO
import Lean
import LeanCopilot.Models.Builtin

open Lean
open LeanCopilot

-- TODO: change?
def configPathEncoderUrl : IO System.FilePath := do
  let home ← IO.getEnv "HOME"
  pure $ (home.getD "/") / ".lean_copilot_current_retriever"

-- TODO: change?
def configPathEmbUrl : IO System.FilePath := do
  let home ← IO.getEnv "HOME"
  pure $ (home.getD "/") / ".lean_copilot_current_emb"

def saveCurrentEncoderUrl (url : String) : IO Unit := do
  let config ← configPathEncoderUrl
  IO.FS.writeFile config url

def saveCurrentEmbUrl (url : String) : IO Unit := do
  let config ← configPathEmbUrl
  IO.FS.writeFile config url

def loadCurrentEncoderUrl : IO String := do
  let config ← configPathEncoderUrl
  if ← config.pathExists then
    let contents ← IO.FS.readFile config
    pure contents.trim
  else
    pure Builtin.encoder.url.toString

def loadCurrentEmbUrl : IO String := do
  let config ← configPathEmbUrl
  if ← config.pathExists then
    let contents ← IO.FS.readFile config
    pure contents.trim
  else
    pure Builtin.premisesUrl.toString

-- TODO: do we need this if we can just download a new url immediately? when we restart the file, this won't be remembered anyway, right?
-- Define a mutable reference to store additional URLs
initialize additionalModelUrlsRef : IO.Ref (List String) ← IO.mkRef []

initialize currentEncoderUrlRef : IO.Ref String ← do
  let url ← loadCurrentEncoderUrl
  IO.mkRef url

initialize currentEmbUrlRef : IO.Ref String ← do
  let url ← loadCurrentEmbUrl
  IO.mkRef url

def builtinModelUrls : List String := [
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-tacgen-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/premise-embeddings-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-byt5-small"
]

-- Function to get all model URLs (built-in + additional)
def getAllModelUrls : IO (List String) := do
  let additional ← additionalModelUrlsRef.get
  for url in additional do
    IO.println s!"Additional URL: {url}"
  return builtinModelUrls ++ additional

def getCurrentEncoderUrl : IO String := currentEncoderUrlRef.get

def getCurrentEmbUrl : IO String := currentEmbUrlRef.get

-- Function to add a new URL to the additional URLs list
-- TODO: optimize by not having separate funcs for ct2 and emb
def addEncoderUrl (url : String) : IO Unit := do
  IO.println "Adding new retriever URL"
  additionalModelUrlsRef.modify (url :: ·)
  let url := Url.parse! url
  IO.println s!"Added new retriever with url {url} and name {url.name!}"

  IO.println "Updating current encoder URL and saving to file"
  currentEncoderUrlRef.set url.toString
  saveCurrentEncoderUrl url.toString
  IO.println "Saved new retriever URL"

  -- TODO: reduce duplication
  -- need to download synchronously to avoid obstructing InfoView output
  IO.println "Downloading new retriever"
  let urls ← getAllModelUrls
  for url in urls do
    IO.println s!"URL: {url}"
  let parsedUrls := Url.parse! <$> urls

  for url in parsedUrls do
    IO.println s!"Downloading {url}"
    try
      downloadUnlessUpToDate url
      IO.println s!"Downloaded {url}"
    catch e =>
      IO.println s!"Error downloading {url}: {e.toString}"

  IO.println "Downloaded all retrievers"

def addEmbUrl (url : String) : IO Unit := do
  IO.println "Adding new embedding URL"
  additionalModelUrlsRef.modify (url :: ·)
  let url := Url.parse! url
  IO.println s!"Added new emb with url {url}"

  IO.println "Updating current embedding URL and saving to file"
  currentEmbUrlRef.set url.toString
  saveCurrentEmbUrl url.toString
  IO.println "Saved new embedding URL"

  -- TODO: reduce duplication
  -- need to download synchronously to avoid obstructing InfoView output
  IO.println "Downloading new embeddings"
  let urls ← getAllModelUrls
  for url in urls do
    IO.println s!"URL: {url}"
  let parsedUrls := Url.parse! <$> urls

  for url in parsedUrls do
    IO.println s!"Downloading {url}"
    try
      downloadUnlessUpToDate url
      IO.println s!"Downloaded {url}"
    catch e =>
      IO.println s!"Error downloading {url}: {e.toString}"

  IO.println "Downloaded all embeddings"

structure Request where
  url : String
deriving ToJson

structure Response where
  output : String
deriving FromJson

def sendUrlTraining {α β : Type} [ToJson α] [FromJson β] (req : α) (url : String) : IO β := do
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

  println! "Done!"
