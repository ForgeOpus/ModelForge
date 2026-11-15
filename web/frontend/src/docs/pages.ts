import {getStarted} from './content/getStarted';
import {basics} from './content/basics';
import {intermediate} from './content/intermediate';
import {advanced} from './content/advanced';
import {reference} from './content/reference';
import type {DocPage} from './ui';

export const PAGES:DocPage[]=[...getStarted,...basics,...intermediate,...advanced,...reference];
