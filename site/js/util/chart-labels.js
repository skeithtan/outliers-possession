/** Place edge labels in separate rows when their text overlaps. */
export function edgeLabelRows(items, left, right, gap) {
  const ends = new Map();
  return items.slice().sort((a, b) => a.x - b.x).map((item) => {
    const half = item.width / 2;
    const x = Math.max(left + half, Math.min(right - half, item.x));
    const rows = ends.get(item.side) || [];
    let row = rows.findIndex((end) => x - half >= end + gap);
    if (row < 0) row = rows.length;
    rows[row] = x + half;
    ends.set(item.side, rows);
    return { ...item, x, row };
  });
}
