(this["webpackJsonppetal-react"]=this["webpackJsonppetal-react"]||[]).push([[0],{75:function(e,t,n){"use strict";n.r(t);var a=n(0),r=n(10),c=n.n(r),o=n(132),s=n(131),i=n(44),l=n(29),u=n(41),h=n(42),j=n(45),f=n(43),p=n(121),b=n(123),d=n(125),m=n(126),g=n(127),O=n(128),v=n(79),y=n(134),x=n(133),S=n(135),w=n(130),I=n(11),C=Object(p.a)({root:{height:"100%"},media:{height:140}});function k(e){var t=C();return Object(I.jsx)(b.a,{className:t.root,children:Object(I.jsx)(d.a,{children:Object(I.jsx)(v.a,{gutterBottom:!0,variant:"h5",component:"h2",children:Object(I.jsx)(m.a,{color:"primary",target:"_blank",rel:"noopener noreferrer",href:e.article.url.S,children:e.article.title.S})})})})}var D=function(e){Object(j.a)(n,e);var t=Object(f.a)(n);function n(){return Object(u.a)(this,n),t.apply(this,arguments)}return Object(h.a)(n,[{key:"render",value:function(){return Object(I.jsx)("div",{})}}]),n}(a.Component),L=function(e){Object(j.a)(n,e);var t=Object(f.a)(n);function n(e){var a;return Object(u.a)(this,n),(a=t.call(this,e)).onSelectionChange=function(e,t){a.setState({selection:t,fetchInProgress:!0},(function(){if(null!=a.state.selection){var e=a.state.selection.id,t=new URL("https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getarticles"),n={level3:e};t.search=new URLSearchParams(n).toString(),fetch(t).then((function(e){return e.json()})).then((function(e){a.setState({fetchInProgress:!1}),a.setState({articlesToDisplay:e.Items})})).catch(console.log)}else a.setState({articlesToDisplay:[],fetchInProgress:!1})}))},a.state={selection:[],functions:[],articlesToDisplay:[]},a.onSelectionChange=a.onSelectionChange.bind(Object(l.a)(a)),a}return Object(h.a)(n,[{key:"render",value:function(){var e=this.state.articlesToDisplay.map((function(e){return Object(I.jsx)(g.a,{item:!0,xs:12,children:Object(I.jsx)(k,{article:e})},e.SortKey.S)}));return Object(I.jsxs)(O.a,{maxWidth:"lg",children:[Object(I.jsxs)(y.a,{my:4,children:[Object(I.jsx)(v.a,{variant:"h4",component:"h1",gutterBottom:!0,children:"How does nature..."}),Object(I.jsx)(S.a,{id:"function",options:this.state.functions.sort((function(e,t){return-t.level2.localeCompare(e.level2)})),groupBy:function(e){return e.level2},getOptionLabel:function(e){return e.level3},style:{width:300,float:"left"},onChange:this.onSelectionChange,renderInput:function(e){return Object(I.jsx)(x.a,Object(i.a)(Object(i.a)({},e),{},{label:"",variant:"outlined"}))}}),this.state.fetchInProgress?Object(I.jsx)(w.a,{style:{float:"left",marginLeft:"20px"}}):Object(I.jsxs)("div",{style:{padding:"20px",float:"left"},children:[this.state.articlesToDisplay.length," results"]})]}),Object(I.jsx)(g.a,{container:!0,spacing:2,direction:"row",justify:"flex-start",alignItems:"stretch",children:e}),Object(I.jsx)(D,{})]})}},{key:"componentDidMount",value:function(){var e=this;fetch("https://ardwrgr7s5.execute-api.us-east-2.amazonaws.com/v1/getalllabels").then((function(e){return e.json()})).then((function(t){var n=[];t.Items.forEach((function(e){n.push({id:e.level3.S.toLowerCase().split(" ").join("_"),level2:e.level2.S,level3:e.level3.S})})),e.setState({functions:n})})).catch(console.log)}}]),n}(a.Component),P=n(53),T=Object(P.a)({typography:{fontFamily:["-apple-system","BlinkMacSystemFont",'"Segoe UI"','"Helvetica Neue"',"Helvetica","Roboto","Arial","sans-serif",'"Apple Color Emoji"','"Segoe UI Emoji"','"Segoe UI Symbol"'].join(",")},palette:{primary:{main:"#9bdaf1"},secondary:{main:"#dd361c"},error:{main:"#dd361c"},type:"dark",background:{}}});c.a.render(Object(I.jsxs)(s.a,{theme:T,children:[Object(I.jsx)(o.a,{}),Object(I.jsx)(L,{})]}),document.querySelector("#root"))}},[[75,1,2]]]);
//# sourceMappingURL=main.635bc5fe.chunk.js.map