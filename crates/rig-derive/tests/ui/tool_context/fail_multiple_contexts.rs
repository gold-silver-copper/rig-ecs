use rig_derive::rig_tool;

struct ToolContext;

#[rig_tool]
fn multiple_contexts(
    #[rig(context)] first: &mut ToolContext,
    #[rig(context)] second: &mut ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = (first, second);
    Ok(())
}

fn main() {}
