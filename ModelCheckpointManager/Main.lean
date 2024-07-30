import ModelCheckpointManager.Url
import ModelCheckpointManager.Download
import Init.System.IO
import Lean

open Lean
open LeanCopilot


def builtinModelUrls : List String := [
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-tacgen-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/premise-embeddings-leandojo-lean4-retriever-byt5-small",
  "https://huggingface.co/kaiyuy/ct2-byt5-small"
]

structure Request where
  text : String
deriving ToJson

structure Response where
  output : String
deriving FromJson

def send {α β : Type} [ToJson α] [FromJson β] (req : α) (url : String) : IO β := do
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
  -- let mut tasks := #[]
  -- let urls := Url.parse! <$> (if args.isEmpty then builtinModelUrls else args)

  -- for url in urls do
  --   tasks := tasks.push $ ← IO.asTask $ downloadUnlessUpToDate url

  -- for t in tasks do
  --   match ← IO.wait t with
  --   | Except.error e => throw e
  --   | Except.ok _ => pure ()

  -- Start progressive training with the initial repository
  -- TODO: ask for url somehow
  IO.println "Starting the program"
  let url := "http://127.0.0.1:8000/reverse/"
  let req : Request := {
    text := "hello"
  }
  -- TODO: make sure the request is sent only once or any request after that is avoided
  let res : Response ← send req url
  IO.println s!"Final response: {res.output}"

  println! "Done!"
