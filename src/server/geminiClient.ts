/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

let clientInstance: GoogleGenAI | null = null;

export function getGeminiClient(): GoogleGenAI {
  if (!clientInstance) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn('[GeminiClient] WARNING: GEMINI_API_KEY is not defined in environment. Fallback simulated responses will be utilized if needed.');
    }
    clientInstance = new GoogleGenAI({
      apiKey: apiKey || 'dummy-dev-key',
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return clientInstance;
}

export const PRIMARY_VISION_MODEL = 'gemini-3.1-flash-lite-preview';
export const FALLBACK_VISION_MODELS = [
  'gemini-3.1-flash-lite-preview',
  'gemini-3-flash-preview',
  'gemini-3.6-flash',
  'gemini-3.5-flash',
  'gemini-2.5-flash',
  'gemini-3.8-flash',
];
