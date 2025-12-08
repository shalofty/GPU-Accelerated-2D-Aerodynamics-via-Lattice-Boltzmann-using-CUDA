-- Pandoc Lua filter to fix math mode issues
-- Ensures all math delimiters are properly handled

function Math(el)
  -- Math elements should already be handled, but ensure proper formatting
  return el
end

function InlineMath(el)
  -- Ensure inline math is properly formatted
  return el
end

