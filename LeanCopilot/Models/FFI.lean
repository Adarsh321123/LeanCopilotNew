import Lean
import LeanCopilot.Models.Interface
import LeanCopilot.Models.Native
import LeanCopilot.Models.Builtin
import ModelCheckpointManager.Main

namespace LeanCopilot

set_option autoImplicit false

namespace FFI


@[extern "is_generator_initialized"]
opaque isGeneratorInitialized : (name : @& String) → Bool

@[extern "is_encoder_initialized"]
opaque isEncoderInitialized : (name : @& String) → Bool

@[extern "init_generator"]
opaque initGenerator (name : @& String) (modelPath : @& String) (computeType : @& String) (device : @& String) (deviceIndex : @& Array UInt64) : Bool

@[extern "init_encoder"]
opaque initEncoder (name : @& String) (modelPath : @& String) (computeType : @& String) (device : @& String) (deviceIndex : @& Array UInt64) : Bool

@[extern "generate"]
opaque generate (name : @& String) (inputTokens : @& Array String) (targetPrefixTokens : @& Array String) (numReturnSequences : UInt64) (beamSize : UInt64)
  (minLength : UInt64) (maxLength : UInt64) (lengthPenalty : Float) (patience : Float) (temperature : Float)
  : Array (Array String × Float)

@[extern "encode"]
opaque encode (name : @& String) (inputTokens : @& Array String) : FloatArray

@[extern "init_premise_embeddings"]
opaque initPremiseEmbeddings (path : @& String) (device : @& String) : Bool

@[extern "premise_embeddings_initialized"]
opaque premiseEmbeddingsInitialized : Unit → Bool

@[extern "init_premise_dictionary"]
opaque initPremiseDictionary (path : @& String) : Bool

@[extern "premise_dictionary_initialized"]
opaque premiseDictionaryInitialized : Unit → Bool

@[extern "retrieve"]
opaque retrieve (queryEmb : @& FloatArray) (k : UInt64) : Array (String × String × String × Float)

@[extern "cuda_available"]
opaque cudaAvailable : Unit → Bool


end FFI


def cudaAvailable : Bool := FFI.cudaAvailable ()


namespace NativeGenerator

-- def joinStrings (arr : Array String) : String :=
--   arr.foldl (fun acc str => acc ++ " " ++ str) ""

def generate (model : NativeGenerator) (input : String) (targetPrefix : String) : IO $ Array (String × Float) := do
  IO.println "Inside generate native generator"
  if ¬ FFI.isGeneratorInitialized model.name then
    let path ← model.path
    IO.println s!"Path: {path}"
    if ¬ (← path.pathExists) then
      throw $ IO.userError s!"Cannot find the model {model.name}. Please run `lake exe download {model.url}`."
    let device := model.device.toString
    IO.println s!"Device: {device}"
    let computeType := model.computeType.toString
    IO.println s!"Compute type: {computeType}"
    if ¬ (FFI.initGenerator model.name path.toString computeType device model.deviceIndex) then
      throw $ IO.userError s!"Failed to initialize model {model.name}"

  let tokenizer := model.tokenizer
  let inputTokens := tokenizer.tokenize input |>.push tokenizer.eosToken
  let targetPrefixTokens := tokenizer.tokenize targetPrefix
  let numReturnSequences := model.params.numReturnSequences
  let beamSize := model.params.beamSize
  let minLength := model.params.minLength
  let maxLength := model.params.maxLength
  let lengthPenalty := model.params.lengthPenalty
  let patience := model.params.patience
  let temperature := model.params.temperature

  -- let tokensWithScores := FFI.generate model.name inputTokens targetPrefixTokens numReturnSequences beamSize minLength maxLength lengthPenalty patience temperature

  IO.println "About to call FFI.generate"
  IO.println s!"Model name: {model.name}"
  let tokensWithScores := FFI.generate model.name inputTokens targetPrefixTokens numReturnSequences beamSize minLength maxLength lengthPenalty patience temperature
  IO.println "FFI call completed successfully"

  IO.println s!"Size: {tokensWithScores.size}"

  let result := tokensWithScores.filterMap fun ((ts, s) : Array String × Float) => (tokenizer.detokenize ts, s)
  IO.println "Result mapped successfully"
  IO.println s!"Number of results: {result.size}"
  return result
  -- return tokensWithScores.filterMap fun ((ts, s) : Array String × Float) => (tokenizer.detokenize ts, s)

  -- IMPT:
  -- try
  --   -- let tokensWithScores := FFI.generate model.name inputTokens targetPrefixTokens numReturnSequences beamSize minLength maxLength lengthPenalty patience temperature

  --   IO.println "About to call FFI.generate"
  --   let tokensWithScores := FFI.generate model.name inputTokens targetPrefixTokens numReturnSequences beamSize minLength maxLength lengthPenalty patience temperature
  --   IO.println "FFI call completed successfully"

  --   IO.println s!"Size: {tokensWithScores.size}"

  --   -- TODO: use original lines below
  --   let result := tokensWithScores.map fun (tokens, score) => (tokens.foldl (· ++ " " ++ ·) "", score)
  --   IO.println "Result mapped successfully"

  --   IO.println s!"Number of results: {result.size}"
  --   return result

  --   -- return #[]
  -- catch e =>
  --   IO.println s!"Exception caught: {e}"
  --   return #[]

  -- IO.println s!"Generate completed with tokensWithScores size: ${tokensWithScores.size}"

  -- if tokensWithScores.isEmpty then
  --   IO.println "No tokens available"
  -- else
  --   IO.println "Tokens are available"

  -- match tokensWithScores with
  -- | #[] => IO.println "The array is empty or not initialized properly."
  -- | _ => IO.println s!"Received tokens with scores"

  -- tokensWithScores.forM fun (tokens, score) => do
  --   let tokensStr := joinStrings tokens -- Join all tokens into a single string
  --   IO.println s!"Tokens: {tokensStr}, Score: {score}"

  -- IO.println tokensWithScores
  -- return tokensWithScores.filterMap fun ((ts, s) : Array String × Float) => (tokenizer.detokenize ts, s)

  -- return #[]


instance : TextToText NativeGenerator where
  generate := NativeGenerator.generate


end NativeGenerator


namespace NativeEncoder


def encode (model : NativeEncoder) (input : String) : IO FloatArray := do
  if ¬ FFI.isEncoderInitialized model.name then
    let path ← model.path
    if ¬ (← path.pathExists) then
      throw $ IO.userError s!"Cannot find the model {model.name}. Please run `lake exe download {model.url}`."
    let device := model.device.toString
    let computeType := model.computeType.toString
    if ¬ (FFI.initEncoder model.name path.toString computeType device model.deviceIndex) then
      throw $ IO.userError s!"Failed to initialize model {model.name}"

  let tokenizer := model.tokenizer
  let inputTokens := tokenizer.tokenize input |>.push tokenizer.eosToken
  return FFI.encode model.name inputTokens


instance : TextToVec NativeEncoder where
  encode := NativeEncoder.encode


end NativeEncoder


def premiseEmbeddingsInitialized : IO Bool := do
  return FFI.premiseEmbeddingsInitialized ()


def initPremiseEmbeddings (device : Device) : Lean.CoreM Bool := do
  let url ← liftM loadCurrentEmbUrl
  IO.println s!"Init premise embeddings with URL: {url}"
  let parsed_url := Url.parse! url
  if ¬(← isUpToDate parsed_url) then
    Lean.logWarning s!"The local premise embeddings are not up to date. You may want to run `lake exe LeanCopilot/download` to re-download it."
  let path := (← getModelDir parsed_url) / "embeddings.npy"
  if ¬ (← path.pathExists) then
    throwError s!"Please run `lake exe download {url}` to download premise embeddings."
    return false
  return FFI.initPremiseEmbeddings path.toString device.toString


def premiseDictionaryInitialized : IO Bool := do
  return FFI.premiseDictionaryInitialized ()


def initPremiseDictionary : IO Bool := do
  let url ← liftM loadCurrentEmbUrl
  IO.println s!"Init premise dictionary with URL: {url}"
  let parsed_url := Url.parse! url
  let path := (← getModelDir parsed_url) / "dictionary.json"
  if ¬ (← path.pathExists) then
    throw $ IO.userError s!"Please run `lake exe download {url}` to download the premise dictionary."
    return false
  return FFI.initPremiseDictionary path.toString


end LeanCopilot
