"""How the explorer page looks: Material theme, stylesheets and component styles."""

# Panel's classic HTML pane otherwise prefers the notebook editor's colors over
# the enclosing Material UI Page. These are Panel's supported design overrides.
MATERIAL_STYLESHEET = """
:host {
  --design-background-color: var(--mui-palette-background-default);
  --design-surface-color: var(--mui-palette-background-paper);
  --design-background-text-color: var(--mui-palette-text-primary);
  --design-surface-text-color: var(--mui-palette-text-primary);
  --design-primary-color: var(--mui-palette-primary-main);
  --design-primary-text-color: var(--mui-palette-primary-contrastText);
}
"""

EXPLORER_THEME = {
    "typography": {
        "fontFamily": "Inter, system-ui, sans-serif",
        "fontSize": 12,
        "button": {"textTransform": "none"},
    },
    "shape": {"borderRadius": 5},
    "components": {
        "MuiToolbar": {
            "styleOverrides": {
                "root": {"minHeight": 44, "@media (min-width:600px)": {"minHeight": 44}}
            }
        },
        "MuiTab": {
            "styleOverrides": {
                "root": {"minHeight": 34, "minWidth": 0, "padding": "6px 14px"}
            }
        },
        "MuiTabs": {"styleOverrides": {"root": {"minHeight": 34}}},
        "MuiSlider": {
            "styleOverrides": {
                "root": {"padding": "10px 0"},
                "thumb": {"width": 12, "height": 12},
            }
        },
        "MuiCardHeader": {
            "styleOverrides": {
                "root": {"padding": "8px 12px"},
                "title": {"fontSize": 13, "fontWeight": 600},
            }
        },
        "MuiCardContent": {
            "styleOverrides": {
                "root": {"padding": 12, "&:last-child": {"paddingBottom": 12}}
            }
        },
    },
}


# Header and sidebar on the paper background, the title in the text colour.
PAGE_STYLE = {
    ".main-content": {"padding": 0},
    ".header": {
        "backgroundColor": "var(--mui-palette-background-paper) !important",
        "color": "text.primary",
        "boxShadow": "none",
        "borderBottom": 1,
        "borderColor": "divider",
    },
    ".header .title": {
        "fontSize": 16,
        "fontWeight": 650,
        "color": "var(--mui-palette-text-primary) !important",
    },
    ".sidebar": {"backgroundColor": "background.paper"},
}


# Two-line names, small secondary text and compact rows in the layer list.
MENU_STYLE = {
    ".MuiListItemText-primary": {
        "overflow": "hidden",
        "textOverflow": "ellipsis",
        "display": "-webkit-box",
        "WebkitLineClamp": 2,
        "WebkitBoxOrient": "vertical",
        "fontSize": 12,
        "fontWeight": 550,
        "lineHeight": 1.4,
    },
    ".MuiListItemText-secondary": {"fontSize": 11},
    ".MuiListItemIcon-root": {"minWidth": 26},
    ".MuiListItemButton-root": {"padding": "4px 8px"},
    ".MuiIconButton-root": {"color": "text.secondary", "padding": "4px"},
    ".material-icons": {"fontSize": 18},
}
