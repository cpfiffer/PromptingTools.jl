# [DRAFT] PromptingTools Compiler + Optimizer

All the below functionality is heavily experimental and untested! Any feedback is appreciated!

It's heavily inspired by DSPy (prompt optimization), Turing.jl (`@model`) and Flux.jl (forward-mode via functors).

**Why?** 
- When chaining multiple steps around AI calls, there starts to be a lot of boilerplate code around logging, unwrapping, retries, asserts etc. We can probably save you some keystrokes!
- If we can define the AI task flow in a more declarative fashion, we can try to optimize it (eg, prompt, parameters, etc.)

**When to use?**
- When you're writing a "program", not just a single AI call 

Downsides: Unnecessary complexity, harder to debug, unnecessary logging (and allocations), ... and many more!

## Compiler
- `@aimodel` provides a very simplistic compiler/DSL for defining a "model", which is a sequence of AI calls defined inside a function
- The motivation is to abstract away common operations (eg, operating on AIMessages vs operating on the generated output) and save a few keystrokes
- The design and syntax is inspired by Turing.jl `@model` and by functors used in Flux.jl
- Practically, `@aimodel` will create a function with empty arguments that upon calling returns an `AIModel`, which is a "lazy" object that contains the AI call logic you defined
- `AIModel` has 8 fields, which are used to store the state of the AI task:
  - `f`: Rewritten function that executes the AI program
  - `traces`: Logs of all the AI conversations
  - `stats`: Statistics about the run
  - ... and a few more housekeeping fields

- When `@aimodel` is called it will:
  - rewrite the ai* calls into their lazy variants (eg, `aigenerate` -> `AIGenerate`)
  - automatically determine if a reference to `ai()` is meant to be `aigenerate` or `aiextract` (depending on the return type annotation), possibly also `aiclassify`
  - separate producing the AI call and extracting the result (create temporary variables for the AICall results vs the (string) output)
  - create tracking of the results (in `aimodel.traces`) and stats (in `aimodel.stats`) for debugging and optimization later on


Available building blocks:
- `@aimodel` rewrites the function definition to create a call-able `AIModel`
- `@airetry` will re-run the last ai* call `max_retries`-times if it fails (no "feedback" is provided) (syntax `@airetry z=ai("...") max_retries=3` or generally `@airetry CODEBLOCK KWARGS`)
- `@aisuggest` soft checks `max_suggests`-times whether the provided output satisfies the condition (syntax: `@aisuggest CODEBLOCK CONDITION FEEDBACK`), if not it will re-run the last ai* call. If it fails more than `max_suggests` times, it will produce a warning but continue execution
  Think of it as a "nudge" to the AI model to produce the desired output/style/format
- `@aiassert` behaves similarly to the `@aisuggest`, except it's a stricter check - if the number of retries is exceeded (`max_asserts`), it will produce an assertion statement and stop the execution (not to waste tokens)
  Think of it as a "guardrail" (eg, check JSON schema, JSON validity)
- `@aiprompt` experimental way to compose the structure of the prompt from the provided building blocks and have the AI model generate it
- `@aieval` has no effect on the AIModel in the `mode=:forwardpass`, it will be used to provide a score in the prompt "optimization" mode

How it works:
```julia
# Define program
@aimodel function my_model(a)
    a=ai("say hi \$x-times"; model="gpt3t")
    return a
end

# Initialize
model = my_model() # initiatives AIModel - no ai* calls made yet
model(3) # triggers the AI program and returns the output directly

# But you can access what happened via:
model.traces[1][:a]
```

The `@aimodel` broadly rewrites the function definition to:
```julia
# aimodel serves as the "memory"
aimodel.f = (aimodel,a) -> begin

      # ... some housekeeping here

      aimodel.callnames[:a] = oyster
      oyster = AIGenerate("say hi $(x)-times"; model = "gpt3t")
      :a ∉ __calls_order && push!(__calls_order, :a)
      run!(oyster)
      # some logging with a lock
      lock(aimodel.lock) do 
          __trace[:a] = oyster.conversation |> copy
          log_call(aimodel, oyster)
      end
      # check for budget, eg, did we exceed maximum number of failed calls or total calls (it's a bit delayed but that's fine)
      check_budget_left(aimodel)
      a = oyster.conversation |> (x->(last(x)).content)

      # ... some housekeeping here
      return a
  end

# by calling the model as a functor, you then call
# model(a) = aimodel.f(model, a)

```


More complicated example:
```julia
@aimodel function my_model(n=2; model="gpt3t")
    # add a soft check for our AI task
    # syntax: @aisuggest CODEBLOCK CONDITION FEEDBACK // or simply @aisuggest CONDITION FEEDBACK if already INSIDE a CODEBLOCK
    @aisuggest begin
        greeting = ai("Say hi $(n)-times"; model)
    end occursin("John", z) "Greeting must include the name John"

    return greeting
end

model = my_model(; max_suggests=2) # initiatives AIModel - no ai* calls made yet
# trigger the AI task like a Flux.jl model
model(3; model="gpt4t")
# Output: ...
```
If the response doesn't include "John" in the output, it will re-run up to 2x times and provide the feedback: "Greetings must include the name John" to the prompt/conversation. If it fails more than 2 times, it will produce a warning but continue execution.


## Optimizer (super raw)
- Optimizer is heavily inspired by [DSPy](https://github.com/stanfordnlp/dspy)
- Under `mode=:optimization`, AIModel changes its behaviour and it will seek to optimize some aspects of your AI task by leveraging the signals from `@aiassert`, `@aisuggest`, and `@aieval` 
- The simplest parameter to optimize is the prompt that is defined declaratively as `@aiprompt` (-> returns `AIPrompt` object)
  - It will sample loosely based on the MCTS algorithm successively improving the blocks defined in the `AIPrompt` (see the example below)
  - At the moment, optimization is sequential and separate for each field depending on its budget


Toy example (not functional):
```julia

prompt=@aiprompt begin
    role = "AI assistant" # rewrite into a full sentence
    task = "question, context -> answer" # or simply say "Your task is to do XYZ..."
    instructions = ["Output: separate by commas"] # to be optimized further
    examples = [("<example context", "<example question>", "<example answer>")] # the optimizer should use these to bootstrap more examples. 
    motivation = "tip $100 if it's correct" # silly tricks to try, optional
    chaiofthought # add the chain of thought phrase
    placeholder_context # insert {{context}} for aicall, will be added a section of System prompt
    placeholder_user_question # insert {{question}} for user inputs, will be added as a section of User prompt
end

```

The hope is to allow drawing on a bank "helpful phrases/tricks", generated examples inspired by provided examples and/or "RAG'd" examples from existing data/databases.