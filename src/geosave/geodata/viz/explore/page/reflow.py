"""Keep a responsive plot's canvas aligned with its Page allocation."""

import panel as pn


class Reflow(pn.custom.JSComponent):
    """Request native layout when Page changes the child's allocated size."""

    object = pn.viewable.Child()

    # Panel 1.10/Bokeh 3.10 can update child bounding boxes without laying out
    # their canvases when the Page root itself has not changed size.
    _esm = """
    export function render({model, view}) {
        const child = model.get_child("object")
        const observer = new ResizeObserver(() => view.compute_layout())
        observer.observe(child)
        model.on("remove", () => observer.disconnect())
        return child
    }
    """
