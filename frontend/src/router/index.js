import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import PrelimView from '../views/PrelimView.vue'
import MoreThan from '../views/MoreThan.vue'
import LessThan from '../views/LessThan.vue'
import ResultView from '../views/ResultView.vue'

const routes = [
  {
    path: '/',
    name: 'HomeView',
    component: HomeView
  },
  {
    path: '/prelim',
    name: 'PrelimView',
    component: PrelimView
  },
  {
    path: '/moreThan',
    name: 'MoreThan',
    component: MoreThan
  },
  {
    path: '/lessThan',
    name: 'LessThan',
    component: LessThan
  },
  {
    path: '/getresult',
    name: 'GetResult',
    component: ResultView
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
