-- LaTeX-specific fixes for the report:
-- * characters the text font (Noto Serif) lacks: math symbols are set in the math font, the
--   hourglass that marks sections waiting for Colab becomes a coloured label;
-- * line-break points after "@" in text (metric names like Recall@100 in narrow table
--   columns) and after "_", "/", "." in inline code (file names).
local replacements = { ["⏳"] = "\\pending{}", ["@"] = "@\\allowbreak{}" }
for _, char in ipairs({ "−", "→", "↔", "≈", "≡", "≤", "≥", "≫", "∈", "∝", "∞", "‖" }) do
  replacements[char] = "\\ensuremath{" .. char .. "}"
end

function Str(el)
  local parts, rest = {}, el.text
  while #rest > 0 do
    local first, last, key = nil, nil, nil
    for char, _ in pairs(replacements) do
      local s, e = rest:find(char, 1, true)
      if s and (not first or s < first) then first, last, key = s, e, char end
    end
    if not first then
      table.insert(parts, pandoc.Str(rest))
      break
    end
    if first > 1 then table.insert(parts, pandoc.Str(rest:sub(1, first - 1))) end
    table.insert(parts, pandoc.RawInline("latex", replacements[key]))
    rest = rest:sub(last + 1)
  end
  return parts
end

local escapes = {
  ["\\"] = "\\textbackslash{}", ["{"] = "\\{", ["}"] = "\\}", ["$"] = "\\$", ["&"] = "\\&",
  ["#"] = "\\#", ["%"] = "\\%", ["^"] = "\\^{}", ["~"] = "\\textasciitilde{}",
  ["_"] = "\\_\\allowbreak{}", ["/"] = "/\\allowbreak{}", ["."] = ".\\allowbreak{}",
}

function Code(el)
  local text = el.text:gsub("[\\{}$&#%%^~_/.]", escapes)
  return pandoc.RawInline("latex", "\\texttt{" .. text .. "}")
end

-- Wide pipe tables get equal column widths from their "|---|" separators; size the columns by
-- the length of their longest cell instead (between 6 and 40 characters), so the method
-- column does not wrap while short number columns stay narrow.
local function cell_length(cell)
  return utf8.len(pandoc.utils.stringify(cell.contents)) or 0
end

function Table(tbl)
  local n = #tbl.colspecs
  if n == 0 or tbl.colspecs[1][2] == nil or tbl.colspecs[1][2] == 0 then
    return nil -- natural widths (narrow table)
  end
  local longest = {}
  for i = 1, n do longest[i] = 6 end
  local function visit(rows)
    for _, row in ipairs(rows) do
      for i, cell in ipairs(row.cells) do
        if i <= n then longest[i] = math.max(longest[i], math.min(cell_length(cell), 40)) end
      end
    end
  end
  visit(tbl.head.rows)
  for _, body in ipairs(tbl.bodies) do visit(body.body) end
  local total = 0
  for i = 1, n do longest[i] = longest[i] + 3 end -- cell padding (\tabcolsep)
  for i = 1, n do total = total + longest[i] end
  for i = 1, n do tbl.colspecs[i][2] = longest[i] / total end
  return tbl
end
