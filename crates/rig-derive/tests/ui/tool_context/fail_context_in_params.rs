use rig_derive::rig_tool;

struct ToolContext;

#[rig_tool(params(context = "obsolete host context"))]
fn context_in_params(
    #[rig(context)] context: &mut ToolContext,
    query: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(query)
}

fn main() {}
