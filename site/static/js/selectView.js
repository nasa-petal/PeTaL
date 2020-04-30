var packSize = [600,600];
var selectOffset = [30,30];

var pack = d3.pack().size(packSize).padding(10);

function updateCircles(root) {
    var dur = 1500;

    var nodes = d3.hierarchy(this.root, function (d) {
        if (!d.expandedCircle || d.children.length === 0) {
            return null;
        }
        return d.children;
    })
        .sum(function (d) {
            if (!d.expandedCircle || d.children.length === 0 ) {
                return d.matches.length;
            }
            return 0;
        })
        .sort(function (a, b) {
            return b.value - a.value;
            // return a.data.name.localeCompare(b.data.name);
        });

    // console.log(root);
    // console.log(nodes);

    pack(nodes);

    var selectBefore = baseSelectGroup.selectAll('circle');

    var u = baseSelectGroup.selectAll('circle').data(nodes.leaves(), d => d.data.id);
    u.each(d => delete d.temppos);

    // give a good starting position to our new circles, centered inside its parent
    var merge = u.enter()
        .each(d => {
            // find the parent
            var p = null;
            selectBefore.each(dd => {
                if(dd.data.id === d.parent.data.id)
                    p = dd;

            });
            if(p == null) {
                d.temppos = [packSize[0]/2,packSize[1]/2];
            }
            else {
                // position it at the tip of the parent's circle, minus this circle's radius
                var dir = normalize(sub([d.x,d.y],[p.x,p.y]));
                var vec = mult(p.r-d.r,dir);
                d.temppos = add(vec,[p.x,p.y]);
            }
            d.temppos = add(d.temppos,selectOffset);

            //d.tempparent = p == null ? {x: packSize[0]/2, y: packSize[1]/2 } : p;
        })
        .insert('circle',':first-child') // insert it at the top of the dom, so it is drawn behind its parent
        .attr('cx',d => d.temppos[0])
        .attr('cy',d => d.temppos[1])
        .attr('r',d => d.r)
        .attr('fill', d => colorMapper(Math.max(0,d.depth-1))) // start it at its parent's color for better blending
        // now, for both the existing and newly entering circles, assign their properties
        .merge(u)
        .on('mousedown', nodeMouseDown)
        .on('mouseup', nodeMouseUp)
        .on('mousemove', nodeMouseMove)
        // .on('mouseover', tip.show) //function(d) { return d.parent == null ? null : tip.show(d)} )
        // .on('mouseout', tip.hide); //function(d) { return d.parent == null ? null : tip.hide(d)} )

    merge
        .transition()
        //.delay(d => d.temppos == null ? 0 : 0)
        .duration(dur)
        .attrs({
            cx: d => d.x + selectOffset[0],
            cy: d => d.y + selectOffset[1],
            r: d => d.r
        })
        .on('end',getSetting("Movement") ? function(d) {
            swayNode(d,d3.select(this)); 
        } : null);

    // after a bit of time has passed, start morphing the color to where it needs to be	
    d3.interval(function() {
        merge.each(function(d) {
            animateColor(d3.select(this), colorMapper(d.depth));
        })
    }, dur * 2/3);

    // u.enter()
    // .each(d => {
    // // find the parent
    // var p = null;
    // selectBefore.each(dd => {
    // if(dd.data.id == d.parent.data.id)
    // p = dd;
    // })			
    // d.tempparent = p == null ? {x: packSize[0]/2, y: packSize[1]/2 } : p;
    // })
    // .insert('circle',':first-child') // insert it at the top of the dom, so it is drawn behind its parent
    // .attr('cx',d => d.tempparent.x)
    // .attr('cy',d => d.tempparent.y)
    // .attr('r',d => d.r)
    // // now, for both the existing and newly entering circles, assign their properties
    // .merge(u)
    // .on('mousedown', nodeClick)
    // .transition()
    // .duration(dur)
    // .attrs({
    // cx: d => d.x,
    // cy: d => d.y,
    // fill: d => colorMapper(d.depth),
    // r: d => d.r
    // })
    // .on('end',getSetting("Movement") ? function(d) { swayNode(d,d3.select(this)); } : null)


    u.exit().transition().duration(dur).attr('r',0).remove();
    //u.exit().remove()
}

// animates a node's fill to a new color
// will operate on a node even if it is being animated via transition() on a different timer
// operates on a dummy object, and sets the attr directly: https://bl.ocks.org/mbostock/5348789
function animateColor(node, newColor) {
    var lock = {};
    d3.select(lock).transition().duration(600)
        .tween("attr:fill", function() {
            var i = d3.interpolateRgb(node.attr('fill'),newColor);
            return function(t) { node.attr('fill',i(t)); }
        })

}

function swayNode(d,node) {
    var center = [d.x + selectOffset[0],d.y + selectOffset[1]];
    //var node = d3.select(this);
    var pos = [node.attr('cx'), node.attr('cy')];
    var angle = Math.random() * Math.PI * 2;
    var vec = [Math.cos(angle),Math.sin(angle)];
    var minDist = 2, maxDist = 10;

    // find a random position around the node's center
    // this will be a vector of length between minDist and maxDist pointed out of the node's center
    var vecDist = Math.random()*(maxDist-minDist) + minDist;
    var newCenter = add(center,mult(vecDist,vec));
    var distMoved = length(sub(pos,newCenter));
    var speed = 3000 / 15; // take 3000 ms to move 15 screen units (e.g.)

    // transition the node from wherever it is to the new position, then loop back and repeat
    node.transition().duration(speed * distMoved).ease(d3.easeLinear)//.ease(d3.easeQuadInOut)
        .attr('cx',newCenter[0])
        .attr('cy',newCenter[1])
        .on('end',function(d) { swayNode(d,node); });
}

function nodeMouseDown(d) {
    itemClicked = d;
}

function nodeMouseMove(d) {
    // spawn a tool to be dragged, but we have to save this tool and let the global mouse events move it around
    // since we will be moving this tool outside the scope of this node
    // if(itemClicked != null && toolBeingDragged == null) {
    //     toolBeingDragged = new PaintTool(d3.mouse(this),[d.data.name],d.data.matches,null,colorMapper(d.depth));
    //     itemClicked = null;
    // }
}

function nodeMouseUp(d) {
    if(toolBeingDragged == null) {
        d.data.expandedCircle = true;
        updateCircles(root);
    }
    itemClicked = null;
}
