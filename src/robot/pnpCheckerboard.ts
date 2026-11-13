/** PnP / 手眼标定用棋盘格参数（与 RobotScene 中棋盘一致） */

/** 方格列数（内角点列数 = COLS - 1） */
export const PNP_CHECKER_COLS = 10;
/** 方格行数（内角点行数 = ROWS - 1） */
export const PNP_CHECKER_ROWS = 7;
/** 每格边长（米） */
export const PNP_CHECKER_SQUARE_M = 0.018;

export type PnpCheckerboardConfig = {
  innerCorners: [number, number];
  squareSizeM: number;
  squaresCols: number;
  squaresRows: number;
};

export function getPnpCheckerboardConfig(): PnpCheckerboardConfig {
  return {
    innerCorners: [PNP_CHECKER_COLS - 1, PNP_CHECKER_ROWS - 1],
    squareSizeM: PNP_CHECKER_SQUARE_M,
    squaresCols: PNP_CHECKER_COLS,
    squaresRows: PNP_CHECKER_ROWS,
  };
}
